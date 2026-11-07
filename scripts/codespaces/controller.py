#!/usr/bin/env python3
"""Trusted-writer hybrid lifecycle controller. Standard library; no remote PAT."""
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time

REPO = 'VictorWong123/tokenomics'
OPS = Path('/workspaces/.tokenomics-operations')
LEASE = Path('/dev/shm/tokenomics-codespace-lifecycle.lease')
RUNNER = Path('/workspaces/.tokenomics-actions-runner')
SESSION = 'tokenomics-actions-listener'
PATH = '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin'
FILES = ('status.json', 'environment.json', 'fixture-pipeline/complete.json',
         'fixture-pipeline/correctness.jsonl', 'fixture-pipeline/requests.jsonl',
         'fixture-pipeline/run_manifest.json')
HASH_FILES = tuple(name for name in FILES if name != 'environment.json')
MAX_FILE, MAX_TOTAL = 16 * 1024**2, 48 * 1024**2
SUMMARY_KEYS = {'sha', 'run_id', 'attempt', 'hostname', 'portable_status', 'blocks',
                'requests', 'correctness_total', 'correctness_passed', 'sha256', 'archive_sha256'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def ident(sha, run, attempt):
    require(bool(re.fullmatch('[a-f0-9]{40}', sha)), 'Invalid source identity')
    require(all(re.fullmatch('[1-9][0-9]{0,19}', x) for x in (run, attempt)), 'Invalid run identity')
    return dict(sha=sha, run_id=run, attempt=attempt)


def current():
    require(os.environ.get('GITHUB_REPOSITORY') == REPO, 'Wrong repository')
    return ident(os.environ['GITHUB_SHA'], os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_RUN_ATTEMPT'])


def runner_identity(name, number):
    require(bool(re.fullmatch('[A-Za-z0-9._-]{1,100}', name)), 'Invalid runner name variable')
    require(bool(re.fullmatch('[1-9][0-9]{0,19}', number)) and number != '2', 'Invalid or retired runner ID')
    return dict(name=name, id=int(number))


def expected_runner():
    return runner_identity(os.environ.get('TOKENOMICS_RUNNER_NAME', ''), os.environ.get('TOKENOMICS_RUNNER_ID', ''))


def root_for(owner):
    return OPS / (owner['run_id'] + '-' + owner['attempt'])


def command(args, timeout=30, data=None, env=None, quiet=False, check=True):
    # External stdout/stderr never reaches a report or Actions log.
    p = subprocess.run(args, input=data, text=True, stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
                       stderr=subprocess.DEVNULL, timeout=timeout, env=env)
    if check:
        require(p.returncode == 0, 'Fixed command failed: ' + Path(args[0]).name)
    require(len(p.stdout or '') <= 1024**2, 'Command output limit')
    return (p.stdout or '').strip() if check else p.returncode


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def mode_owner(info, mode, directory=False):
    require((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == mode, 'Unsafe ownership or mode')


def lease_verify(owner):
    mode_owner(LEASE.lstat(), 0o600)
    require(json.loads(LEASE.read_text()) == owner, 'Lease owner mismatch')


def marker_verify(owner):
    root = root_for(owner)
    mode_owner(root.lstat(), 0o700, directory=True)
    marker = root / 'owner.json'
    require(not marker.is_symlink() and json.loads(marker.read_text()) == owner, 'Operation marker mismatch')
    return root


def ownership(owner):
    lease_verify(owner)
    return marker_verify(owner)


def acquire(owner):
    require(command(['stat', '-f', '-c', '%T', '/dev/shm']) == 'tmpfs', 'Lease requires tmpfs')
    require(not OPS.is_symlink(), 'Unsafe operations root')
    OPS.mkdir(mode=0o700, exist_ok=True)
    mode_owner(OPS.lstat(), 0o700, directory=True)
    fd = os.open(LEASE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        mode_owner(os.fstat(stream.fileno()), 0o600)
        json.dump(owner, stream)
        stream.flush()
        os.fsync(stream.fileno())
    root = root_for(owner)
    root.mkdir(mode=0o700)  # Never adopt an existing operation.
    (root / 'owner.json').write_text(json.dumps(owner))
    return root


def registered_runner(expected):
    mode_owner(RUNNER.lstat(), 0o700, directory=True)
    config = RUNNER / '.runner'
    require(not config.is_symlink() and config.is_file() and config.stat().st_uid == os.getuid(), 'Missing runner metadata')
    info = json.loads(config.read_text(encoding='utf-8-sig'))
    require(info.get('gitHubUrl') == 'https://github.com/' + REPO and
            type(info.get('agentId')) is int and info['agentId'] == expected['id'] and
            info.get('agentName') == expected['name'], 'Registered runner identity mismatch')
    return info


def processes(kind):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / 'cmdline').read_bytes().split(b'\0')
            # Match argv basename, not snippets of this helper passed on stdin.
            matches = [part.decode(errors='replace') for part in raw
                       if Path(part.decode(errors='replace')).name == kind]
            if matches:
                require(entry.stat().st_uid == os.getuid(), 'Foreign runner process')
                found.append((int(entry.name), matches))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def no_workers(timeout=60):
    deadline = time.monotonic() + timeout
    while processes('Runner.Worker'):
        require(time.monotonic() < deadline, 'Runner worker still active; refusing cleanup/stop')
        time.sleep(2)


def listener_sessions():
    result = subprocess.run(['tmux', 'list-sessions', '-F', '#S'], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    if result.returncode:
        # tmux uses 1 for an absent server; other failures are ambiguous, not empty.
        absent = result.returncode == 1 and (
            result.stderr.startswith('no server running on ') or
            (result.stderr.startswith('error connecting to ') and '(No such file or directory)' in result.stderr))
        require(absent, 'Cannot verify tmux session set')
        return set()
    require(len(result.stdout) <= 4096, 'Tmux session output limit')
    names = result.stdout.splitlines()
    require(names and all(names) and len(set(names)) == len(names), 'Ambiguous tmux session set')
    return set(names)


def process_descends_from(pid, ancestor):
    for _ in range(64):
        info = Path('/proc') / str(pid)
        require(info.stat().st_uid == os.getuid(), 'Foreign process ancestry')
        if pid == ancestor:
            return True
        # comm may contain spaces/parentheses: split after its final closing paren.
        tail = (info / 'stat').read_text().rpartition(') ')[2].split()
        require(len(tail) >= 2, 'Ambiguous listener ancestry')
        parent = int(tail[1])
        if parent <= 1 or parent == pid:
            return False
        pid = parent
    return False


def session_contains_listener(listener_pid):
    panes = command(['tmux', 'list-panes', '-s', '-t', '=' + SESSION,
                     '-F', '#{pane_pid}:#{pane_dead}']).splitlines()
    require(len(panes) == 1 and re.fullmatch('[1-9][0-9]*:0', panes[0]),
            'Owned tmux pane identity ambiguous')
    require(process_descends_from(listener_pid, int(panes[0].split(':')[0])),
            'Current listener is not in owned tmux session')


def start_listener(owner, expected):
    root = ownership(owner)
    registered_runner(expected)
    require(not processes('Runner.Worker') and not processes('Runner.Listener'), 'Existing runner process conflict')
    # Dedicated account/machine: do not attach to or destroy any existing tmux session.
    require(not listener_sessions(), 'Existing tmux session conflict')
    log = root / 'listener.log'
    # Bind readiness to this start: never consume an old registration log.
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        mode_owner(os.fstat(stream.fileno()), 0o600)
        stream.truncate(0)
        stream.flush()
    shell = 'exec ./run.sh > ' + shlex.quote(str(log)) + ' 2>&1'
    command(['tmux', 'new-session', '-d', '-s', SESSION, '-c', str(RUNNER), shell])
    (root / 'listener-owner.json').write_text(json.dumps(owner))
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        ownership(owner)
        listeners = processes('Runner.Listener')
        if log.is_file() and log.stat().st_size <= 1024**2:
            if ('Listening for Jobs' in log.read_text(errors='replace') and len(listeners) == 1
                    and all(path.startswith(str(RUNNER) + '/') for path in listeners[0][1])):
                return
        time.sleep(1)
    raise RuntimeError('Listener readiness not confirmed')


def clean_listener(owner, expected):
    root = ownership(owner)
    registered_runner(expected)
    no_workers()
    # Refuse to signal a process outside the dedicated registered runner path.
    listeners = processes('Runner.Listener')
    require(all(all(path.startswith(str(RUNNER) + '/') for path in paths) for _, paths in listeners),
            'Foreign listener; refusing stop')
    listener_marker = root / 'listener-owner.json'
    started = (listener_marker.is_file() and not listener_marker.is_symlink()
               and json.loads(listener_marker.read_text()) == owner)
    sessions = listener_sessions()
    if not started:
        require(not listeners and not sessions, 'Pre-existing session; refusing cleanup/stop')
    else:
        require(sessions == {SESSION}, 'Foreign or missing tmux session; refusing cleanup/stop')
        require(len(listeners) == 1, 'Current listener identity ambiguous')
        session_contains_listener(listeners[0][0])
        ownership(owner)
        no_workers(timeout=1)
        require(listener_sessions() == {SESSION}, 'Tmux session set changed; refusing cleanup/stop')
        command(['tmux', 'kill-session', '-t', '=' + SESSION])
    deadline = time.monotonic() + 15
    while processes('Runner.Listener'):
        require(time.monotonic() < deadline, 'Listener did not exit')
        time.sleep(1)
    no_workers(timeout=1)
    require(not listener_sessions(), 'Tmux sessions remain; refusing cleanup/stop')
    ownership(owner)
    shutil.rmtree(root)
    lease_verify(owner)  # Never unlink: it fences every contender until Shutdown.
    return {'retained_owner': owner}


def remote(phase, owner, expected):
    if phase == 'initialize':
        acquire(owner)
        start_listener(owner, expected)
        return {'ready_owner': owner}
    require(phase == 'cleanup', 'Invalid remote phase')
    return clean_listener(owner, expected)


def target(space, name, shutdown=False):
    require(space.get('name') == name and space.get('repository', {}).get('full_name') == REPO
            and space['repository'].get('private') is True and space.get('machine', {}).get('cpus') == 2
            and space.get('idle_timeout_minutes') == 240, 'Target metadata mismatch')
    if shutdown:
        require(space.get('state') == 'Shutdown', 'Target not Shutdown; refusing overlap')


def ssh_args(name, phase, owner, expected):
    require(phase in ('initialize', 'cleanup'), 'Invalid SSH phase')
    fixed = ['/usr/bin/python3', '-', '--remote', phase, owner['sha'], owner['run_id'], owner['attempt'],
             expected['name'], str(expected['id'])]
    return ['gh', 'codespace', 'ssh', '--codespace', name, '--', shlex.join(fixed)]


def remote_call(name, phase, owner, expected):
    text = command(ssh_args(name, phase, owner, expected), timeout=120, data=Path(__file__).read_text())
    lines = [line[7:] for line in text.splitlines() if line.startswith('RESULT ')]
    require(len(lines) == 1, 'Missing remote protocol result')
    result = json.loads(lines[0])
    require(result.get('ok') is True, 'Remote lifecycle phase failed')
    return result['result']


def space_get(name):
    return json.loads(command(['gh', 'api', '/user/codespaces/' + name]))


def stop_confirm(name, timeout=90):
    deadline = time.monotonic() + timeout
    command(['gh', 'codespace', 'stop', '--codespace', name], timeout=20, check=False)
    while time.monotonic() < deadline:
        try:
            if space_get(name).get('state') == 'Shutdown':
                return
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            pass
        time.sleep(2)
    raise RuntimeError('Actual Shutdown not confirmed')


def finalize_owned(name, owner, expected):
    space = space_get(name)
    target(space, name)
    if space.get('state') == 'Shutdown':
        return  # Never SSH-start an already stopped target.
    require(space.get('state') == 'Available', 'Unexpected lifecycle state; refusing SSH/stop')
    proof = remote_call(name, 'cleanup', owner, expected)
    require(proof.get('retained_owner') == owner, 'Retained ownership not confirmed')
    stop_confirm(name)


def fixture_environment():
    require(command(['/usr/bin/python3', '-c', 'import sys; print("%d.%d" % sys.version_info[:2])']) == '3.12',
            'System Python must be 3.12')
    command(['/usr/bin/python3', '-c', 'import venv, ensurepip'])
    return dict(os.environ, PATH=PATH, TOKENOMICS_BUILD_JOBS='2')


def manifest(owner, hostname):
    require(bool(re.fullmatch('[A-Za-z0-9._-]{1,128}', hostname)), 'Invalid hostname')
    return dict(owner, hostname=hostname, repository=REPO, evidence_files=list(FILES))


def archive_create(results, path, proof):
    total = 0
    with tarfile.open(path, 'w:gz') as tar:
        for name in FILES:
            source = results / name
            require(source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(results.resolve()),
                    'Missing or unsafe evidence')
            size = source.stat().st_size
            total += size
            require(size <= MAX_FILE and total <= MAX_TOTAL, 'Evidence size limit')
            tar.add(source, arcname=name, recursive=False)
        data = json.dumps(proof, sort_keys=True).encode()
        member = tarfile.TarInfo('execution-manifest.json')
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))


def archive_validate(path, proof):
    require(path.stat().st_size <= MAX_TOTAL, 'Archive size limit')
    parsed, hashes = {}, {}
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers()
        require(len(members) == len(FILES) + 1 and {m.name for m in members} == set(FILES) | {'execution-manifest.json'},
                'Archive member mismatch')
        require(all(m.isfile() and 0 <= m.size <= MAX_FILE for m in members)
                and sum(m.size for m in members) <= MAX_TOTAL, 'Unsafe archive metadata')
        for member in members:
            raw = tar.extractfile(member).read()
            hashes[member.name] = hashlib.sha256(raw).hexdigest()
            parsed[member.name] = ([json.loads(line) for line in raw.splitlines()] if member.name.endswith('.jsonl')
                                   else json.loads(raw))
    require(parsed['execution-manifest.json'] == proof, 'Archive source mismatch')
    status, complete = parsed['status.json'], parsed['fixture-pipeline/complete.json']
    require(isinstance(status, dict) and isinstance(complete, dict), 'Malformed producer status')
    require(status.get('state') == 'portable_checks_passed' and type(status.get('records')) is int and status['records'] == 168,
            'Portable fixture status mismatch')
    require(type(complete.get('blocks')) is int and complete['blocks'] == 24
            and type(complete.get('records')) is int and complete['records'] == 168, 'Producer counts mismatch')
    requests, checks = parsed['fixture-pipeline/requests.jsonl'], parsed['fixture-pipeline/correctness.jsonl']
    require(len(requests) == len(checks) == 168 and all(isinstance(r, dict) for r in requests + checks), 'Producer record mismatch')
    require(all(r.get('ok') is True for r in checks), 'Correctness gate failed')
    for rows in (requests, checks):
        counts = collections.Counter(row.get('block_id') for row in rows)
        require(len(counts) == 24 and all(isinstance(k, str) and v == 7 for k, v in counts.items()), 'Producer block layout mismatch')
    require(isinstance(parsed['fixture-pipeline/run_manifest.json'], dict), 'Malformed producer manifest')
    summary = {key: proof[key] for key in ('sha', 'run_id', 'attempt', 'hostname')}
    summary.update(portable_status='portable_checks_passed', blocks=24, requests=168,
                   correctness_total=168, correctness_passed=168,
                   sha256={name: hashes[name] for name in HASH_FILES}, archive_sha256=digest(path))
    summary_validate(summary, {k: proof[k] for k in ('sha', 'run_id', 'attempt')})
    return summary


def summary_validate(summary, owner):
    require(isinstance(summary, dict) and set(summary) == SUMMARY_KEYS, 'Summary key mismatch')
    require(all(summary.get(k) == v for k, v in owner.items()), 'Summary source mismatch')
    manifest(owner, summary['hostname'])
    require(summary['portable_status'] == 'portable_checks_passed', 'Summary status mismatch')
    for key, value in [('blocks', 24), ('requests', 168), ('correctness_total', 168), ('correctness_passed', 168)]:
        require(type(summary[key]) is int and summary[key] == value, 'Summary count mismatch')
    require(isinstance(summary['sha256'], dict) and set(summary['sha256']) == set(HASH_FILES), 'Summary digest keys mismatch')
    require(all(isinstance(v, str) and re.fullmatch('[a-f0-9]{64}', v)
                for v in [summary['archive_sha256'], *summary['sha256'].values()]), 'Summary digest mismatch')
    return summary


def output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
        stream.write(name + '=' + value + '\n')


def workload(owner, expected, artifacts):
    root = ownership(owner)
    registered_runner(expected)
    require(os.environ.get('RUNNER_NAME') == expected['name'], 'Wrong workload runner')
    workspace = Path(os.environ['GITHUB_WORKSPACE']).resolve()
    require(not workspace.is_relative_to(Path('/workspaces/tokenomics')) and not Path('/workspaces/tokenomics').is_relative_to(workspace),
            'User checkout cannot be workload workspace')
    require(command(['git', '-C', str(workspace), 'rev-parse', 'HEAD']) == owner['sha'], 'Workload checkout SHA mismatch')
    env = fixture_environment()
    results = root / 'results'
    require(not results.exists(), 'Results already exist')
    command(['timeout', '--signal=TERM', '--kill-after=15s', '12000s', 'bash', str(workspace / 'run.sh'),
             'cpu', '--fixtures-only', '--threads', '2', '--run-dir', str(results)], timeout=12030, env=env, quiet=True)
    ownership(owner)
    proof = manifest(owner, socket.gethostname())
    archive = artifacts / 'evidence.tar.gz'
    archive_create(results, archive, proof)
    summary = archive_validate(archive, proof)
    output('summary', json.dumps(summary, sort_keys=True, separators=(',', ':')))
    print('TOKENOMICS_WORKLOAD_SUMMARY ' + json.dumps(summary, sort_keys=True))
    return summary


def lifecycle(mode, owner, expected):
    require(bool(os.environ.get('GH_TOKEN')), 'Missing lifecycle secret')
    name = os.environ.get('TOKENOMICS_CODESPACE', '')
    require(bool(re.fullmatch('[a-z0-9][a-z0-9-]{0,99}', name)), 'Invalid target variable')
    if mode == 'initialize':
        output('claimed', 'false')
        target(space_get(name), name, shutdown=True)
        try:
            ready = remote_call(name, 'initialize', owner, expected)
            require(ready.get('ready_owner') == owner, 'Listener ownership not confirmed')
            output('claimed', 'true')
        except Exception:
            # Even if readiness/transport failed, only exact remotely verified ownership
            # can authorize cleanup and stop. A lease denial cannot stop its owner.
            try:
                finalize_owned(name, owner, expected)
            except Exception:
                pass
            raise
        return {'claimed': True}
    finalize_owned(name, owner, expected)
    # Lifecycle success is independent of any workload-specific evidence contract.
    # Missing/failed/custom workloads must never bypass owned shutdown cleanup.
    try:
        summary = json.loads(os.environ.get('WORKLOAD_SUMMARY', ''))
        summary_validate(summary, owner)
    except (ValueError, TypeError, KeyError, RuntimeError):
        return {'shutdown_confirmed': True, 'fixture_evidence_valid': False,
                'evidence_status': 'No valid fixture summary; lifecycle cleanup only'}
    final = dict(summary, codespace=name, machine_cpus=2, lease_filesystem='tmpfs', shutdown_confirmed=True)
    print('TOKENOMICS_EVIDENCE_SUMMARY ' + json.dumps(final, sort_keys=True))
    return {'shutdown_confirmed': True, 'fixture_evidence_valid': True}


def main(mode):
    # Only fixed fields are reported on any failure; external errors are suppressed.
    report = {'mode': mode, 'passed': False}
    artifacts = Path(os.environ['RUNNER_TEMP']) / 'tokenomics-artifacts'
    artifacts.mkdir(exist_ok=True)
    try:
        owner = current()
        expected = expected_runner()
        report.update(owner)
        if mode == 'workload':
            report['summary'] = workload(owner, expected, artifacts)
        else:
            report.update(lifecycle(mode, owner, expected))
        report['passed'] = True
    except Exception as ex:
        report['error_type'] = type(ex).__name__
        # All RuntimeError messages originate from fixed checks above.
        if isinstance(ex, RuntimeError):
            report['error'] = str(ex)
        if mode == 'workload':
            (artifacts / 'evidence.tar.gz').unlink(missing_ok=True)
    (artifacts / 'report.json').write_text(json.dumps(report, sort_keys=True) + '\n')
    print('TOKENOMICS_CONTROLLER_REPORT ' + json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    if len(sys.argv) == 8 and sys.argv[1] == '--remote':
        try:
            owner = ident(*sys.argv[3:6])
            expected = runner_identity(*sys.argv[6:8])
            result = remote(sys.argv[2], owner, expected)
            print('RESULT ' + json.dumps({'ok': True, 'result': result}))
        except Exception:
            print('RESULT ' + json.dumps({'ok': False}))
    elif len(sys.argv) == 2 and sys.argv[1] in ('initialize', 'workload', 'finalize'):
        sys.exit(main(sys.argv[1]))
    else:
        sys.exit('Fixed controller mode required')
