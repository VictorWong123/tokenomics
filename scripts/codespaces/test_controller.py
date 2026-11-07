import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('controller', Path(__file__).with_name('controller.py'))
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)
OWNER = c.ident('a' * 40, '123', '2')
OTHER = c.ident('b' * 40, '456', '1')
EXPECTED = c.runner_identity('new-runner', '22')
REPO_ROOT = Path(__file__).resolve().parents[2]


class HybridTests(unittest.TestCase):
    def space(self, state='Shutdown', **extra):
        result = dict(name='target', repository=dict(full_name=c.REPO, private=True),
                      machine=dict(cpus=2), idle_timeout_minutes=240, state=state)
        result.update(extra)
        return result

    def env(self, root):
        return dict(GITHUB_REPOSITORY=c.REPO, GITHUB_SHA=OWNER['sha'], GITHUB_RUN_ID='123',
                    GITHUB_RUN_ATTEMPT='2', RUNNER_TEMP=str(root), GITHUB_OUTPUT=str(root / 'outputs'),
                    GH_TOKEN='DO-NOT-EMIT', TOKENOMICS_CODESPACE='target',
                    TOKENOMICS_RUNNER_NAME=EXPECTED['name'], TOKENOMICS_RUNNER_ID='22')

    def registration(self, root):
        runner = root / 'runner'
        runner.mkdir(mode=0o700)
        info = dict(agentName=EXPECTED['name'], agentId=22, gitHubUrl='https://github.com/' + c.REPO)
        (runner / '.runner').write_text('\ufeff' + json.dumps(info))
        return runner, info

    def summary(self):
        return dict(OWNER, hostname='test-host', portable_status='portable_checks_passed', blocks=24,
                    requests=168, correctness_total=168, correctness_passed=168,
                    sha256={name: 'c' * 64 for name in c.HASH_FILES}, archive_sha256='d' * 64)

    def cached_producer(self, root):
        # Real bench.run aggregation from cached immutable blocks, no binary/build/network.
        sys.path.insert(0, str(REPO_ROOT))
        import bench.run as producer
        binary = root / 'dummy'
        binary.write_bytes(b'cached only; must never execute')
        results = root / 'results'
        out = results / 'fixture-pipeline'
        configs = [{'id': x} for x in ('target', 'resident', 'fixed2', 'fixed4', 'confidence', 'acceptance', 'upstream-style')]
        prompts = [{'id': f'development-{cat}-{n:02d}'} for cat in ('qa', 'code', 'summary') for n in (0, 3)]
        cfg = dict(configurations=configs, repetitions=2, output_caps=[32, 128], seed=1)
        for prompt, cap, rep, order in producer.balanced_blocks(prompts, [32, 128], 2, configs, 1):
            block_id = f"{prompt['id']}-n{cap}-r{rep}"
            block = out / 'blocks' / block_id
            block.mkdir(parents=True)
            names = []
            for conf in order:
                name = conf['id'] + '.json'
                (block / name).write_text(json.dumps(dict(configuration_id=conf['id'], block_id=block_id,
                                                        prompt='RAW-PROMPT-NEVER-EMIT')))
                names.append(name)
            (block / 'complete.json').write_text(json.dumps(dict(records=names,
                correctness=[dict(ok=True, block_id=block_id) for _ in names],
                sha256={name: producer.sha256(block / name) for name in names})))
        with patch.object(producer, 'hardware', return_value={}), \
                patch.object(producer, 'source_fingerprint', return_value={}), \
                patch.object(producer.subprocess, 'run', side_effect=AssertionError('No binary allowed')):
            rows = producer.run(binary, binary, None, prompts, cfg, out)
        (results / 'status.json').write_text(json.dumps(dict(state='portable_checks_passed', records=len(rows))))
        (results / 'environment.json').write_text('{"hardware":"DO-NOT-EMIT"}')
        return results

    def test_workflow_structure_secret_isolation_and_legacy_manual_only(self):
        import yaml
        workflow = yaml.load((REPO_ROOT / '.github/workflows/codespace-fixtures.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(workflow['on'], {'push': {'branches': ['work', 'codex/**']}})
        self.assertEqual(workflow['permissions'], {'contents': 'read'})
        self.assertEqual(workflow['concurrency'], {'group': 'tokenomics-dedicated-codespace', 'cancel-in-progress': 'false'})
        jobs = workflow['jobs']
        self.assertEqual(set(jobs), {'initialize', 'workload', 'finalize'})
        for name in ('initialize', 'finalize'):
            self.assertEqual(jobs[name]['runs-on'], 'ubuntu-24.04')
            self.assertEqual(jobs[name]['timeout-minutes'], '5')
            self.assertEqual(jobs[name]['steps'][0]['with'], {'ref': '${{ github.sha }}', 'persist-credentials': 'false'})
            self.assertIn('secrets.CODESPACES_TOKEN', json.dumps(jobs[name]))
        self.assertEqual(jobs['workload']['runs-on'], ['self-hosted', 'linux', 'x64', 'tokenomics-codespace-long-tasks'])
        self.assertEqual(jobs['workload']['timeout-minutes'], '210')
        self.assertEqual(jobs['workload']['needs'], 'initialize')
        self.assertEqual(jobs['workload']['if'], "needs.initialize.outputs.claimed == 'true'")
        self.assertNotIn('CODESPACES_TOKEN', json.dumps(jobs['workload']))
        self.assertNotIn('GH_TOKEN', json.dumps(jobs['workload']))
        checkout = jobs['workload']['steps'][0]['with']
        self.assertEqual(checkout, {'ref': '${{ github.sha }}', 'clean': 'true', 'fetch-depth': '0',
                                    'submodules': 'recursive', 'persist-credentials': 'false'})
        self.assertEqual(jobs['finalize']['needs'], ['initialize', 'workload'])
        self.assertEqual(jobs['finalize']['if'], "${{ always() && needs.initialize.outputs.claimed == 'true' }}")
        self.assertEqual(jobs['workload']['steps'][-1]['if'], 'always()')
        for job in jobs.values():
            for step in job['steps']:
                if 'uses' in step:
                    self.assertRegex(step['uses'], r'@[a-f0-9]{40}$')
        legacy = yaml.load((REPO_ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(legacy['on'], ['workflow_dispatch'])
        self.assertEqual(legacy['jobs']['macos']['runs-on'], 'macos-15')
        old = subprocess.check_output(['git', 'show', '87cb30a8:.github/workflows/ci.yml'], cwd=REPO_ROOT, text=True)
        self.assertEqual((REPO_ROOT / '.github/workflows/ci.yml').read_text(),
                         old.replace('on: [push, pull_request]', 'on: [workflow_dispatch]').replace('runs-on: macos-14', 'runs-on: macos-15'))
        self.assertNotIn('PATCH', (REPO_ROOT / 'scripts/codespaces/controller.py').read_text())

    def test_target_admission_is_exact_including_creation_idle_timeout(self):
        c.target(self.space(), 'target', shutdown=True)
        for change in (dict(state='Available'), dict(idle_timeout_minutes=30), dict(machine={'cpus': 4}),
                       dict(name='other'), dict(repository=dict(full_name=c.REPO, private=False))):
            with self.assertRaises(RuntimeError):
                c.target(self.space(**change), 'target', shutdown=True)

    def test_invalid_identity_and_old_registration_rejected(self):
        for args in [('$(id)', '123', '2'), ('a' * 40, '../x', '2')]:
            with self.assertRaises(RuntimeError):
                c.ident(*args)
        with self.assertRaises(RuntimeError):
            c.runner_identity('old-runner', '2')
        args = c.ssh_args('target', 'initialize', OWNER, EXPECTED)
        self.assertEqual(shlex.split(args[-1]), ['/usr/bin/python3', '-', '--remote', 'initialize', OWNER['sha'], '123', '2', 'new-runner', '22'])
        self.assertNotIn('GH_TOKEN', args[-1])

    def test_runner_metadata_bom_exact_repo_name_id_and_directory_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, info = self.registration(root)
            with patch.object(c, 'RUNNER', runner):
                self.assertEqual(c.registered_runner(EXPECTED)['agentId'], 22)
                for key, value in [('agentId', 2), ('agentName', 'other'), ('gitHubUrl', 'https://github.com/other/repo')]:
                    (runner / '.runner').write_text(json.dumps(dict(info, **{key: value})))
                    with self.assertRaises(RuntimeError):
                        c.registered_runner(EXPECTED)
                (runner / '.runner').write_text(json.dumps(info))
                runner.chmod(0o755)
                with self.assertRaises(RuntimeError):
                    c.registered_runner(EXPECTED)

    def test_lifetime_lease_race_metadata_and_no_unlink_after_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, _ = self.registration(root)
            with patch.object(c, 'OPS', root / 'ops'), patch.object(c, 'LEASE', root / 'lease'), \
                    patch.object(c, 'RUNNER', runner), patch.object(c, 'command', return_value='tmpfs'), \
                    patch.object(c, 'processes', return_value=[]), patch.object(c, 'listener_sessions', return_value=set()):
                c.acquire(OWNER)
                self.assertEqual(c.LEASE.stat().st_mode & 0o777, 0o600)
                with self.assertRaises(FileExistsError):
                    c.acquire(OTHER)
                with self.assertRaises(RuntimeError):
                    c.clean_listener(OTHER, EXPECTED)
                self.assertEqual(c.clean_listener(OWNER, EXPECTED), {'retained_owner': OWNER})
                self.assertFalse(c.root_for(OWNER).exists())
                c.lease_verify(OWNER)
                with self.assertRaises(FileExistsError):
                    c.acquire(OTHER)
                c.LEASE.chmod(0o644)
                with self.assertRaises(RuntimeError):
                    c.lease_verify(OWNER)
                with self.assertRaises(RuntimeError):
                    c.mode_owner(type('Stat', (), {'st_mode': stat.S_IFREG | 0o600, 'st_uid': os.getuid() + 1})(), 0o600)

    def test_non_tmpfs_and_symlink_lease_denied(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OPS', Path(tmp) / 'ops'), \
                patch.object(c, 'LEASE', Path(tmp) / 'lease'):
            with patch.object(c, 'command', return_value='ext4'):
                with self.assertRaises(RuntimeError):
                    c.acquire(OWNER)
            c.LEASE.symlink_to(Path(tmp) / 'foreign')
            with patch.object(c, 'command', return_value='tmpfs'):
                with self.assertRaises(FileExistsError):
                    c.acquire(OWNER)
            with self.assertRaises(RuntimeError):
                c.lease_verify(OWNER)

    def test_initialize_claim_only_after_listener_ready_and_no_pat_in_ssh(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self.env(Path(tmp))), \
                patch.object(c, 'space_get', return_value=self.space()), \
                patch.object(c, 'remote_call', return_value={'ready_owner': OWNER}) as remote, \
                patch.object(c, 'output') as output:
            self.assertEqual(c.lifecycle('initialize', OWNER, EXPECTED), {'claimed': True})
            remote.assert_called_once_with('target', 'initialize', OWNER, EXPECTED)
            self.assertEqual([x.args for x in output.call_args_list], [('claimed', 'false'), ('claimed', 'true')])

    def test_failed_init_cleanup_requires_exact_ownership_and_nonowner_never_stops(self):
        for owned in (True, False):
            remote_results = [RuntimeError('Readiness failed'), {'retained_owner': OWNER} if owned else RuntimeError('Lease mismatch')]
            with self.subTest(owned=owned), tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, self.env(Path(tmp))), \
                    patch.object(c, 'space_get', side_effect=[self.space(), self.space('Available')]), \
                    patch.object(c, 'remote_call', side_effect=remote_results), patch.object(c, 'stop_confirm') as stop, \
                    patch.object(c, 'output') as output:
                with self.assertRaises(RuntimeError):
                    c.lifecycle('initialize', OWNER, EXPECTED)
                self.assertEqual(stop.call_count, int(owned))
                self.assertEqual([x.args for x in output.call_args_list], [('claimed', 'false')])

    def test_finalize_already_shutdown_does_not_ssh_or_stop(self):
        with patch.object(c, 'space_get', return_value=self.space()), patch.object(c, 'remote_call') as remote, \
                patch.object(c, 'stop_confirm') as stop:
            c.finalize_owned('target', OWNER, EXPECTED)
            remote.assert_not_called()
            stop.assert_not_called()

    def test_finalize_refuses_nonowner_unexpected_state_and_active_worker(self):
        for state in ('Available', 'Starting'):
            with patch.object(c, 'space_get', return_value=self.space(state)), \
                    patch.object(c, 'remote_call', return_value={'retained_owner': OTHER}), patch.object(c, 'stop_confirm') as stop:
                with self.assertRaises(RuntimeError):
                    c.finalize_owned('target', OWNER, EXPECTED)
                stop.assert_not_called()
        with patch.object(c, 'processes', return_value=[(1, [])]), patch.object(c.time, 'monotonic', side_effect=[0, 1]):
            with self.assertRaisesRegex(RuntimeError, 'still active'):
                c.no_workers(timeout=0)

    def test_foreign_tmux_conflict_is_never_killed_or_stopped(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'OPS', Path(tmp) / 'ops'), \
                patch.object(c, 'LEASE', Path(tmp) / 'lease'), patch.object(c, 'command', return_value='tmpfs'), \
                patch.object(c, 'registered_runner'), patch.object(c, 'processes', return_value=[]), \
                patch.object(c, 'listener_sessions', return_value={'foreign'}):
            c.acquire(OWNER)
            with self.assertRaisesRegex(RuntimeError, 'Existing tmux'):
                c.start_listener(OWNER, EXPECTED)
            with self.assertRaisesRegex(RuntimeError, 'Pre-existing session'):
                c.clean_listener(OWNER, EXPECTED)
            c.lease_verify(OWNER)


    def test_owned_tmux_start_fresh_log_and_exact_listener_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, _ = self.registration(root)
            active = [False]
            def invoke(args, **kwargs):
                if args[0] == 'stat':
                    return 'tmpfs'
                if args[:2] == ['tmux', 'list-panes']:
                    return '100:0'
                if args[:2] == ['tmux', 'new-session']:
                    log = c.root_for(OWNER) / 'listener.log'
                    self.assertEqual(log.read_text(), '')  # Old readiness was truncated before launch.
                    self.assertNotIn(str(runner / 'listener.log'), args[-1])
                    log.write_text('Listening for Jobs\n')
                    active[0] = True
                    return ''
                if args[:2] == ['tmux', 'has-session']:
                    return 0
                if args[:2] == ['tmux', 'kill-session']:
                    self.assertEqual(args[-1], '=' + c.SESSION)
                    active[0] = False
                    return ''
                self.fail('Unexpected local command')
            def processes(kind):
                return [(123, [str(runner / 'bin/Runner.Listener')])] if active[0] and kind == 'Runner.Listener' else []
            with patch.object(c, 'OPS', root / 'ops'), patch.object(c, 'LEASE', root / 'lease'), \
                    patch.object(c, 'RUNNER', runner), patch.object(c, 'command', side_effect=invoke), \
                    patch.object(c, 'processes', side_effect=processes), \
                    patch.object(c, 'listener_sessions', side_effect=lambda: {c.SESSION} if active[0] else set()), \
                    patch.object(c, 'process_descends_from', return_value=True):
                c.acquire(OWNER)
                stale = c.root_for(OWNER) / 'listener.log'
                stale.write_text('Listening for Jobs\n')
                stale.chmod(0o600)
                c.start_listener(OWNER, EXPECTED)
                self.assertEqual(json.loads((c.root_for(OWNER) / 'listener-owner.json').read_text()), OWNER)
                self.assertTrue(active[0])
                self.assertEqual(c.clean_listener(OWNER, EXPECTED), {'retained_owner': OWNER})
                self.assertFalse(active[0])
                c.lease_verify(OWNER)
                self.assertFalse(c.root_for(OWNER).exists())


    def test_late_foreign_tmux_and_post_kill_session_refuse_codespace_stop(self):
        for session_sets, expected_kills in (([{c.SESSION, 'interactive'}], 0),
                                             ([{c.SESSION}, {c.SESSION}, {'interactive'}], 1)):
            with self.subTest(sessions=session_sets), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runner, _ = self.registration(root)
                killed = [False]
                def invoke(args, **kwargs):
                    if args[0] == 'stat':
                        return 'tmpfs'
                    if args[:2] == ['tmux', 'kill-session']:
                        killed[0] = True
                        return ''
                    self.fail('Unexpected command')
                def processes(kind):
                    return [(123, [str(runner / 'bin/Runner.Listener')])] if kind == 'Runner.Listener' and not killed[0] else []
                with patch.object(c, 'OPS', root / 'ops'), patch.object(c, 'LEASE', root / 'lease'), \
                        patch.object(c, 'RUNNER', runner), patch.object(c, 'command', side_effect=invoke), \
                        patch.object(c, 'processes', side_effect=processes), \
                        patch.object(c, 'listener_sessions', side_effect=session_sets), \
                        patch.object(c, 'session_contains_listener'), \
                        patch.object(c, 'space_get', return_value=self.space('Available')), \
                        patch.object(c, 'remote_call', side_effect=lambda *args: c.clean_listener(OWNER, EXPECTED)), \
                        patch.object(c, 'stop_confirm') as stop:
                    c.acquire(OWNER)
                    (c.root_for(OWNER) / 'listener-owner.json').write_text(json.dumps(OWNER))
                    with self.assertRaises(RuntimeError):
                        c.finalize_owned('target', OWNER, EXPECTED)
                    stop.assert_not_called()
                    self.assertEqual(int(killed[0]), expected_kills)
                    self.assertTrue(c.root_for(OWNER).exists())
                    c.lease_verify(OWNER)

    def test_tmux_listener_must_match_single_current_pane_ancestry(self):
        for panes, descendant in (('100:0\n101:0', True), ('100:1', True), ('100:0', False)):
            with patch.object(c, 'command', return_value=panes), \
                    patch.object(c, 'process_descends_from', return_value=descendant):
                with self.assertRaises(RuntimeError):
                    c.session_contains_listener(123)
        with patch.object(c, 'command', return_value='100:0'), \
                patch.object(c, 'process_descends_from', return_value=True) as ancestry:
            c.session_contains_listener(123)
            ancestry.assert_called_once_with(123, 100)

    def test_tmux_session_enumeration_fails_closed_on_unknown_error(self):
        for code, stdout, stderr, expected in (
                (1, '', 'no server running on /tmp/tmux-1000/default', set()),
                (0, 'owned\ninteractive\n', '', {'owned', 'interactive'}),
                (1, '', 'permission denied', None)):
            result = type('Result', (), dict(returncode=code, stdout=stdout, stderr=stderr))()
            with patch.object(c.subprocess, 'run', return_value=result):
                if expected is None:
                    with self.assertRaises(RuntimeError):
                        c.listener_sessions()
                else:
                    self.assertEqual(c.listener_sessions(), expected)

    def test_stale_readiness_without_new_listener_evidence_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, _ = self.registration(root)
            with patch.object(c, 'OPS', root / 'ops'), patch.object(c, 'LEASE', root / 'lease'), \
                    patch.object(c, 'RUNNER', runner), patch.object(c, 'command', return_value='tmpfs'), \
                    patch.object(c, 'processes', return_value=[]), patch.object(c, 'listener_sessions', return_value=set()):
                c.acquire(OWNER)
                log = c.root_for(OWNER) / 'listener.log'
                log.write_text('Listening for Jobs\n')
                log.chmod(0o600)
                with patch.object(c.time, 'monotonic', side_effect=[0, 46]):
                    with self.assertRaisesRegex(RuntimeError, 'readiness not confirmed'):
                        c.start_listener(OWNER, EXPECTED)
                self.assertEqual(log.read_text(), '')

    def test_archive_extra_and_link_members_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.tar.gz'
            proof = c.manifest(OWNER, 'test-host')
            for link in (True, False):
                with tarfile.open(path, 'w:gz') as tar:
                    for name in (*c.FILES, 'execution-manifest.json'):
                        data = json.dumps(proof if name == 'execution-manifest.json' else {}).encode()
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        if link and name == 'status.json':
                            member.type = tarfile.SYMTYPE
                            member.linkname = '/etc/passwd'
                            member.size = 0
                        tar.addfile(member, io.BytesIO(data))
                    if not link:
                        member = tarfile.TarInfo('../unexpected')
                        member.size = 0
                        tar.addfile(member, io.BytesIO())
                with self.assertRaises(RuntimeError):
                    c.archive_validate(path, proof)

    def test_stop_poll_requires_actual_shutdown(self):
        with patch.object(c, 'command', return_value=1), \
                patch.object(c, 'space_get', side_effect=[self.space('ShuttingDown'), self.space()]), \
                patch.object(c.time, 'sleep'):
            c.stop_confirm('target')
        with patch.object(c, 'command', return_value=0), patch.object(c.time, 'monotonic', side_effect=[0, 100]):
            with self.assertRaisesRegex(RuntimeError, 'Shutdown not confirmed'):
                c.stop_confirm('target')

    def test_real_producer_24_168_archive_exact_summary_and_no_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = self.cached_producer(root)
            self.assertEqual({p.name for p in (results / 'fixture-pipeline').iterdir() if p.is_file()},
                             {'run_manifest.json', 'complete.json', 'requests.jsonl', 'correctness.jsonl'})
            archive = root / 'evidence.tar.gz'
            proof = c.manifest(OWNER, 'test-host')
            c.archive_create(results, archive, proof)
            summary = c.archive_validate(archive, proof)
            self.assertEqual(set(summary), c.SUMMARY_KEYS)
            self.assertEqual(set(summary['sha256']), set(c.HASH_FILES))
            self.assertEqual((summary['blocks'], summary['requests'], summary['correctness_passed']), (24, 168, 168))
            self.assertNotIn('RAW-PROMPT', json.dumps(summary))
            self.assertNotIn('DO-NOT-EMIT', json.dumps(summary))
            with tarfile.open(archive) as tar:
                self.assertEqual({m.name for m in tar.getmembers()}, set(c.FILES) | {'execution-manifest.json'})
            for change in (dict(requests=True), dict(prompt='unexpected'), dict(sha='b' * 40), dict(archive_sha256='bad')):
                with self.assertRaises(RuntimeError):
                    c.summary_validate(dict(summary, **change), OWNER)

    def test_failed_producer_archive_content_never_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = self.cached_producer(root)
            archive = root / 'evidence.tar.gz'
            proof = c.manifest(OWNER, 'test-host')
            checks = results / 'fixture-pipeline/correctness.jsonl'
            original = checks.read_text()
            checks.write_text(original.replace('true', 'false', 1))
            c.archive_create(results, archive, proof)
            with self.assertRaises(RuntimeError):
                c.archive_validate(archive, proof)
            checks.write_text(original)
            (results / 'fixture-pipeline/complete.json').write_text('{"blocks":24,"records":167}')
            c.archive_create(results, archive, proof)
            with self.assertRaises(RuntimeError):
                c.archive_validate(archive, proof)
            (results / 'status.json').unlink()
            (results / 'status.json').symlink_to(root / 'dummy')
            with self.assertRaises(RuntimeError):
                c.archive_create(results, archive, proof)

    def test_finalize_summary_only_after_shutdown_including_cancelled_workload(self):
        for stop_ok, summary in ((True, self.summary()), (False, self.summary()), (True, {})):
            with self.subTest(stop_ok=stop_ok, summary=bool(summary)), tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, dict(self.env(Path(tmp)), WORKLOAD_SUMMARY=json.dumps(summary))), \
                    patch.object(c, 'space_get', return_value=self.space('Available')), \
                    patch.object(c, 'remote_call', return_value={'retained_owner': OWNER}), \
                    patch.object(c, 'stop_confirm', side_effect=None if stop_ok else RuntimeError('Shutdown not confirmed')) as stop, \
                    patch('builtins.print') as emit:
                expected = stop_ok and bool(summary)
                self.assertEqual(c.main('finalize'), 0 if stop_ok else 1)
                stop.assert_called_once_with('target')  # Also for missing/cancelled workload summary.
                lines = [x.args[0] for x in emit.call_args_list if x.args[0].startswith('TOKENOMICS_EVIDENCE_SUMMARY ')]
                self.assertEqual(len(lines), int(expected))
                self.assertNotIn('DO-NOT-EMIT', str(emit.call_args_list))
                self.assertTrue((Path(tmp) / 'tokenomics-artifacts/report.json').exists())

    def test_workload_real_archive_to_summary_and_failed_artifact_path(self):
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                results = self.cached_producer(root)
                operation = root / 'operation'
                operation.mkdir()
                results.rename(operation / 'results')
                # Workload command creates results; fake only the fixture execution,
                # then use real archive creation, validation and output protocol.
                generated = operation / 'generated'
                (operation / 'results').rename(generated)
                workspace = root / 'workspace'
                workspace.mkdir()
                def invoke(args, **kwargs):
                    if args[0] == 'git':
                        return OWNER['sha']
                    if not success:
                        raise RuntimeError('Fixed fixture command failed')
                    generated.rename(operation / 'results')
                    self.assertEqual(kwargs['env']['PATH'], c.PATH)
                    self.assertEqual(kwargs['env']['TOKENOMICS_BUILD_JOBS'], '2')
                    self.assertIn('--fixtures-only', args)
                    return ''
                env = dict(self.env(root), GITHUB_WORKSPACE=str(workspace), RUNNER_NAME=EXPECTED['name'])
                env.pop('GH_TOKEN')
                with patch.dict(os.environ, env, clear=True), patch.object(c, 'ownership', return_value=operation), \
                        patch.object(c, 'registered_runner'), patch.object(c, 'command', side_effect=invoke), \
                        patch.object(c, 'fixture_environment', return_value=dict(PATH=c.PATH, TOKENOMICS_BUILD_JOBS='2')), \
                        patch('builtins.print') as emit:
                    self.assertEqual(c.main('workload'), 0 if success else 1)
                artifacts = root / 'tokenomics-artifacts'
                self.assertEqual((artifacts / 'evidence.tar.gz').exists(), success)
                self.assertTrue((artifacts / 'report.json').exists())
                self.assertNotIn('RAW-PROMPT', str(emit.call_args_list))


    def test_custom_long_workload_cleanup_is_not_fixture_or_duration_gated(self):
        for value in ('', 'not-json', '{}'):
            order = []
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,
                    dict(self.env(Path(tmp)), WORKLOAD_SUMMARY=value)), \
                    patch.object(c, 'finalize_owned', side_effect=lambda *args: order.append('shutdown')), \
                    patch.object(c, 'summary_validate', side_effect=RuntimeError('Not fixture evidence')), \
                    patch('builtins.print') as emit:
                result = c.lifecycle('finalize', OWNER, EXPECTED)
                self.assertTrue(result['shutdown_confirmed'])
                self.assertFalse(result['fixture_evidence_valid'])
                self.assertEqual(order, ['shutdown'])
                self.assertFalse(emit.called)
        import inspect
        source = inspect.getsource(c.finalize_owned)
        self.assertNotIn('archive_validate', source)
        self.assertNotIn('summary_validate', source)
        self.assertNotIn('24', source)
        workload = inspect.getsource(c.workload)
        self.assertIn("'12000s'", workload)  # 200m wrapper, not obsolete 20m.
        self.assertNotIn("'1200s'", workload)

    def test_python_preflight_selects_system_interpreter(self):
        with patch.object(c, 'command', side_effect=['3.12', '']) as invoke:
            env = c.fixture_environment()
            self.assertEqual(env['PATH'], '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin')
            self.assertTrue(all(call.args[0][0] == '/usr/bin/python3' for call in invoke.call_args_list))
        with patch.object(c, 'command', return_value='3.14'):
            with self.assertRaises(RuntimeError):
                c.fixture_environment()


if __name__ == '__main__':
    unittest.main()
