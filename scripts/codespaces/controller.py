#!/usr/bin/env python3
"""Bounded, trusted-writer Codespace fixture controller; no third-party modules."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time

REPO = 'VictorWong123/tokenomics'
OPS = Path('/workspaces/.tokenomics-operations')
FILES = ('status.json', 'environment.json', 'fixture-pipeline/complete.json',
         'fixture-pipeline/correctness.jsonl', 'fixture-pipeline/cycles.jsonl',
         'fixture-pipeline/requests.jsonl', 'fixture-pipeline/run_manifest.json')
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 48 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def identity(sha, run, attempt):
    require(bool(re.fullmatch('[0-9a-f]{40}', sha)), 'Invalid SHA')
    require(all(re.fullmatch('[1-9][0-9]{0,19}', s) for s in (run, attempt)), 'Invalid run identity')
    return {'sha': sha, 'run_id': run, 'attempt': attempt}


def target(space, name):
    require(space.get('name') == name and space.get('repository', {}).get('full_name') == REPO
            and space['repository'].get('private') is True
            and space.get('machine', {}).get('cpus') == 2, 'Target identity mismatch')
    require(space.get('state') == 'Shutdown', 'Target not Shutdown; refusing overlap')


def command(args, timeout=60, data=None, env=None, quiet=False):
    # Never relay arbitrary process output/errors (which may contain credentials).
    p = subprocess.run(args, input=data, text=True, stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout, env=env)
    require(p.returncode == 0, 'Command failed: ' + Path(args[0]).name)
    require(len(p.stdout or '') <= 1024 * 1024, 'Command output too large')
    return (p.stdout or '').strip()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(ident, hostname, bundle_digest):
    require(bool(re.fullmatch('[0-9a-f]{64}', bundle_digest)), 'Invalid bundle digest')
    require(bool(re.fullmatch('[A-Za-z0-9._-]{1,128}', hostname)), 'Invalid hostname')
    return dict(ident, hostname=hostname, repository=REPO, workspace=str(OPS / key(ident) / 'tree'),
                bundle_sha256=bundle_digest, evidence_files=list(FILES))


def key(ident):
    return ident['run_id'] + '-' + ident['attempt']


def ssh_args(name, phase, ident, bundle_digest):
    require(phase in ('prepare', 'run', 'clean'), 'Invalid phase')
    args = ['/usr/bin/python3', '-', '--remote', phase, ident['sha'], ident['run_id'], ident['attempt'], bundle_digest]
    return ['gh', 'codespace', 'ssh', '--codespace', name, '--', shlex.join(args)]


def stop_and_confirm(name, timeout=180):
    deadline = time.monotonic() + timeout
    try:
        command(['gh', 'codespace', 'stop', '--codespace', name], timeout=min(45, timeout))
    except (RuntimeError, subprocess.TimeoutExpired):
        pass  # Actual state, not stop's exit status, decides success.
    while time.monotonic() < deadline:
        try:
            state = json.loads(command(['gh', 'api', '/user/codespaces/' + name],
                                      timeout=min(20, deadline - time.monotonic())))
            if state.get('state') == 'Shutdown':
                return
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            pass
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    raise RuntimeError('Actual Shutdown not confirmed')


def validate_archive(path, expected_digest, expected):
    require(path.stat().st_size <= MAX_TOTAL, 'Archive too large')
    require(digest(path) == expected_digest, 'Archive digest mismatch')
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers()
        require(len(members) == len(FILES) + 1 and
                {m.name for m in members} == set(FILES) | {'execution-manifest.json'}, 'Unexpected archive content')
        require(all(m.isfile() and 0 <= m.size <= MAX_FILE for m in members)
                and sum(m.size for m in members) <= MAX_TOTAL, 'Unsafe archive member')
        for m in members:
            data = tar.extractfile(m).read()
            if m.name.endswith('.jsonl'):
                for line in data.splitlines():
                    json.loads(line)
            else:
                parsed = json.loads(data)
                if m.name == 'execution-manifest.json':
                    require(parsed == expected, 'Manifest mismatch')
                if m.name == 'status.json':
                    require(parsed.get('state') == 'portable_checks_passed', 'Fixtures did not pass')



def fixture_environment():
    require(command(['/usr/bin/python3', '-c',
                     'import sys; print("%d.%d" % sys.version_info[:2])']) == '3.12',
            'Remote /usr/bin/python3 must be Python 3.12')
    try:
        command(['/usr/bin/python3', '-c', 'import venv, ensurepip'])
    except RuntimeError:
        raise RuntimeError('Remote Python 3.12 venv support unavailable') from None
    return dict(os.environ, PATH='/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin', TOKENOMICS_BUILD_JOBS='2')



def verify_lease(ident):
    lease = OPS / 'lease.json'
    require(not lease.is_symlink() and lease.is_file()
            and json.loads(lease.read_text()) == ident, 'Remote lease owner mismatch')
    return lease


def acquire_lease(ident):
    # O_EXCL is the persistent ownership boundary across separate SSH phases.
    lease = OPS / 'lease.json'
    with lease.open('x') as stream:
        json.dump(ident, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return lease


def remote(phase, ident, bundle_digest):
    import fcntl
    require(bool(re.fullmatch('[0-9a-f]{64}', bundle_digest)), 'Invalid bundle digest')
    require(not OPS.is_symlink(), 'Unsafe operations root')
    OPS.mkdir(mode=0o700, exist_ok=True)
    require(OPS.stat().st_uid == os.getuid(), 'Operations root owner mismatch')
    require(not (OPS / 'controller.lock').is_symlink(), 'Unsafe operation lock')
    with open(OPS / 'controller.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return remote_owned(phase, ident, bundle_digest)


def remote_owned(phase, ident, bundle_digest):
    root = OPS / key(ident)
    marker = root / 'owner.json'
    if phase == 'prepare':
        origin = command(['git', '-C', '/workspaces/tokenomics', 'remote', 'get-url', 'origin'])
        require(origin in ('https://github.com/' + REPO + '.git', 'https://github.com/' + REPO,
                           'git@github.com:' + REPO + '.git'), 'User checkout origin mismatch')
        require(not root.exists() and not root.is_symlink(), 'Operation already exists')
        acquire_lease(ident)
        try:
            root.mkdir(mode=0o700)
            marker.write_text(json.dumps(ident))
        except Exception:
            verify_lease(ident).unlink()
            raise
        return {'prepared': True, 'lease_owner': ident}
    verify_lease(ident)
    require(not root.is_symlink() and not marker.is_symlink() and marker.is_file() and json.loads(marker.read_text()) == ident,
            'Owned operation marker mismatch')
    if phase == 'clean':
        shutil.rmtree(root)  # Only the exact uniquely created, marker-verified operation.
        verify_lease(ident).unlink()
        return {'cleaned': True, 'released_owner': ident}
    require(phase == 'run', 'Invalid remote phase')
    bundle = root / 'source.bundle'
    require(bundle.is_file() and not bundle.is_symlink() and digest(bundle) == bundle_digest,
            'Source bundle digest mismatch')
    source, tree = root / 'source.git', root / 'tree'
    command(['git', 'clone', '--bare', str(bundle), str(source)], timeout=120)
    command(['git', '--git-dir=' + str(source), 'bundle', 'verify', str(bundle)])
    require(command(['git', '--git-dir=' + str(source), 'rev-parse', 'HEAD']) == ident['sha'],
            'Bundle HEAD mismatch')
    command(['git', '--git-dir=' + str(source), 'worktree', 'add', '--detach', str(tree), ident['sha']])
    try:
        require(command(['git', '-C', str(tree), 'rev-parse', 'HEAD']) == ident['sha'], 'Worktree SHA mismatch')
        # Only this pinned public dependency may initialize; reject extra/private submodules.
        modules = tree / '.gitmodules'
        require(command(['git', 'config', '-f', str(modules), '--get-regexp', 'submodule.*.url'])
                == 'submodule.third_party/llama.cpp.url https://github.com/ggml-org/llama.cpp.git',
                'Unexpected submodule URLs')
        require(command(['git', 'config', '-f', str(modules), '--get-regexp', 'submodule.*.path'])
                == 'submodule.third_party/llama.cpp.path third_party/llama.cpp', 'Unexpected submodule paths')
        env = fixture_environment()
        # timeout kills the fixture process group before remote worktree cleanup.
        command(['timeout', '--signal=TERM', '--kill-after=15s', '1200s',
                 'bash', str(tree / 'run.sh'), 'cpu', '--fixtures-only', '--threads', '2',
                 '--run-dir', 'results/local/codespace-fixtures'], timeout=1230, env=env, quiet=True)
        out = tree / 'results/local/codespace-fixtures'
        proof = manifest(ident, socket.gethostname(), bundle_digest)
        archive = root / 'evidence.tar.gz'
        total = 0
        with tarfile.open(archive, 'w:gz') as tar:
            for name in FILES:
                path = out / name
                require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(out.resolve()),
                        'Missing or unsafe evidence')
                total += path.stat().st_size
                require(path.stat().st_size <= MAX_FILE and total <= MAX_TOTAL, 'Evidence too large')
                tar.add(path, arcname=name, recursive=False)
            data = json.dumps(proof, sort_keys=True).encode()
            info = tarfile.TarInfo('execution-manifest.json')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return {'manifest': proof, 'archive_sha256': digest(archive)}
    finally:
        command(['git', '--git-dir=' + str(source), 'worktree', 'remove', '--force', '--force', str(tree)], timeout=60)


def controller():
    report = {'passed': False, 'admitted': False, 'claimed': False, 'remote_prepared': False, 'shutdown_confirmed': False}
    output = Path(os.environ['RUNNER_TEMP']) / 'tokenomics-controller'
    output.mkdir(exist_ok=True)
    log = output / 'controller.log'
    source = Path(__file__).read_text()
    name = os.environ.get('TOKENOMICS_CODESPACE', '')
    ident = None
    bundle_digest = None
    def event(message):
        with log.open('a') as f:
            f.write(message + '\n')
    def remote_call(phase, timeout=90):
        raw = command(ssh_args(name, phase, ident, bundle_digest), timeout=timeout, data=source)
        lines = [line[7:] for line in raw.splitlines() if line.startswith('RESULT ')]
        require(len(lines) == 1, 'Missing unique remote result')
        value = json.loads(lines[0])
        require(value.get('ok') is True, 'Remote phase failed: ' + phase + ': ' + value.get('error', value.get('error_type', 'unknown')))
        return value['result']
    try:
        require(bool(os.environ.get('GH_TOKEN')) and bool(name), 'Missing token or target variable')
        require(bool(re.fullmatch('[a-z0-9][a-z0-9-]{0,99}', name)), 'Invalid target variable')
        require(os.environ.get('GITHUB_REPOSITORY') == REPO, 'Wrong repository')
        ident = identity(os.environ['GITHUB_SHA'], os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_RUN_ATTEMPT'])
        report.update(ident)
        require(command(['git', 'rev-parse', 'HEAD']) == ident['sha'], 'Controller checkout SHA mismatch')
        bundle = Path(os.environ['RUNNER_TEMP']) / ('tokenomics-' + key(ident) + '.bundle')
        command(['git', 'bundle', 'create', str(bundle), 'HEAD'], timeout=120)
        bundle_digest = digest(bundle)
        report['bundle_sha256'] = bundle_digest
        event('Exact-source bundle created')
        target(json.loads(command(['gh', 'api', '/user/codespaces/' + name])), name)
        report['admitted'] = True
        event('Stopped dedicated Codespace admitted; lease not yet acquired')
        prepared = remote_call('prepare')
        require(prepared.get('lease_owner') == ident, 'Lease acquisition not confirmed')
        report['claimed'] = True
        report['remote_prepared'] = True
        event('Exact remote lease acquired')
        command(['gh', 'codespace', 'cp', '--codespace', name, str(bundle),
                 'remote:' + str(OPS / key(ident) / 'source.bundle')], timeout=180)
        proof = remote_call('run', timeout=1500)
        expected = manifest(ident, proof['manifest']['hostname'], bundle_digest)
        require(proof['manifest'] == expected, 'Remote manifest mismatch')
        archive = output / 'evidence.tar.gz'
        command(['gh', 'codespace', 'cp', '--codespace', name,
                 'remote:' + str(OPS / key(ident) / 'evidence.tar.gz'), str(archive)], timeout=120)
        validate_archive(archive, proof['archive_sha256'], expected)
        report.update(manifest=expected, archive_sha256=proof['archive_sha256'], passed=True)
        event('Fixture evidence copied and verified')
    except Exception as ex:
        # Only our fixed RuntimeError messages; never external process output.
        report['error_type'] = type(ex).__name__
        if isinstance(ex, RuntimeError):
            report['error'] = str(ex)
        event('Controller failed: ' + type(ex).__name__)
    finally:
        if report['claimed']:
            try:
                released = remote_call('clean', timeout=90)
                require(released.get('released_owner') == ident, 'Lease release not confirmed')
                report['lease_released'] = True
                event('Owned operation removed and exact lease released')
            except Exception as ex:
                report.update(passed=False, cleanup_error=type(ex).__name__)
                event('Lease cleanup not confirmed; refusing stop')
            if report.get('lease_released'):
                try:
                    stop_and_confirm(name)
                    report['shutdown_confirmed'] = True
                    event('Actual Shutdown confirmed')
                except Exception as ex:
                    report.update(passed=False, stop_error=type(ex).__name__)
        if not report['passed']:
            (output / 'evidence.tar.gz').unlink(missing_ok=True)
        (output / 'controller-report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--remote':
        try:
            ident = identity(*sys.argv[3:6])
            result = remote(sys.argv[2], ident, sys.argv[6])
            print('RESULT ' + json.dumps({'ok': True, 'result': result}))
        except Exception as ex:
            failure = {'ok': False, 'error_type': type(ex).__name__}
            if isinstance(ex, RuntimeError):
                failure['error'] = str(ex)
            print('RESULT ' + json.dumps(failure))
            # Transport success delivers the structured failure to the controller.
            sys.exit(0)
    else:
        sys.exit(controller())
