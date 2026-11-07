import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('controller', Path(__file__).with_name('controller.py'))
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)
SHA = 'a' * 40
DIGEST = 'b' * 64
IDENT = c.identity(SHA, '123', '2')


class ControllerTests(unittest.TestCase):
    def space(self, **changes):
        value = dict(name='target', repository=dict(full_name=c.REPO, private=True),
                     machine=dict(cpus=2), state='Shutdown')
        value.update(changes)
        return value

    def test_admission_rejects_overlap_and_wrong_identity(self):
        c.target(self.space(), 'target')
        for value in (self.space(state='Available'), self.space(name='other'),
                      self.space(machine=dict(cpus=4)),
                      self.space(repository=dict(full_name=c.REPO, private=False)),
                      self.space(repository=dict(full_name='other/repo', private=True))):
            with self.assertRaises(RuntimeError):
                c.target(value, 'target')

    def test_identity_rejects_shell_input(self):
        for args in [('$(id)', '123', '2'), (SHA, '../1', '2'), (SHA, '123', '0')]:
            with self.assertRaises(RuntimeError):
                c.identity(*args)
        with self.assertRaises(RuntimeError):
            c.manifest(IDENT, 'host', 'not-a-digest')

    def test_fixed_ssh_command_has_no_token_or_remote_origin_fetch(self):
        args = c.ssh_args('target', 'run', IDENT, DIGEST)
        self.assertEqual(args[:6], ['gh', 'codespace', 'ssh', '--codespace', 'target', '--'])
        self.assertIn('/usr/bin/python3 - --remote run ' + SHA + ' 123 2 ' + DIGEST, args[-1])
        self.assertNotIn('GH_TOKEN', args[-1])
        with self.assertRaises(RuntimeError):
            c.ssh_args('target', 'arbitrary', IDENT, DIGEST)


    def test_remote_python_preflight_and_controlled_path(self):
        with patch.object(c, 'command', side_effect=['3.12', '']) as run:
            env = c.fixture_environment()
            self.assertEqual(env['PATH'], '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin')
            self.assertEqual(env['TOKENOMICS_BUILD_JOBS'], '2')
            self.assertTrue(all(call.args[0][0] == '/usr/bin/python3' for call in run.call_args_list))
            self.assertIn('import venv, ensurepip', run.call_args_list[1].args[0])
        with patch.object(c, 'command', return_value='3.14'):
            with self.assertRaisesRegex(RuntimeError, 'must be Python 3.12'):
                c.fixture_environment()
        with patch.object(c, 'command', side_effect=['3.12', RuntimeError('Command failed: python3')]):
            with self.assertRaises(RuntimeError):
                c.fixture_environment()

    def test_bundle_preserves_real_commit_and_excludes_git_credentials(self):
        # Local temporary git repositories only; no network or lifecycle calls.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
            git('init', '-q')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.invalid')
            (root / 'source.txt').write_text('fixture\n')
            git('add', 'source.txt')
            git('commit', '-qm', 'source')
            sha = git('rev-parse', 'HEAD')
            git('config', 'http.extraheader', 'sentinel-not-transferred')
            bundle = root / 'source.bundle'
            git('bundle', 'create', str(bundle), 'HEAD')
            git('clone', '--bare', str(bundle), str(root / 'copy.git'))
            self.assertEqual(git('--git-dir=copy.git', 'rev-parse', 'HEAD'), sha)
            self.assertNotIn('sentinel-not-transferred', (root / 'copy.git/config').read_text())
            self.assertNotIn(b'sentinel-not-transferred', bundle.read_bytes())

    def archive(self, path, expected, extra=False, symlink=False):
        with tarfile.open(path, 'w:gz') as tar:
            for name in (*c.FILES, 'execution-manifest.json') + (('../escape',) if extra else ()):
                data = json.dumps(expected if name == 'execution-manifest.json' else
                                  {'state': 'portable_checks_passed'}).encode() + b'\n'
                info = tarfile.TarInfo(name)
                info.size = len(data)
                if symlink and name == 'status.json':
                    info.type = tarfile.SYMTYPE
                    info.linkname = '/etc/passwd'
                    info.size = 0
                tar.addfile(info, io.BytesIO(data))

    def test_evidence_exact_manifest_digest_names_and_types(self):
        expected = c.manifest(IDENT, 'host', DIGEST)
        self.assertEqual(expected['workspace'], '/workspaces/.tokenomics-operations/123-2/tree')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evidence.tar.gz'
            self.archive(path, expected)
            c.validate_archive(path, c.digest(path), expected)
            with self.assertRaises(RuntimeError):
                c.validate_archive(path, '0' * 64, expected)
            with self.assertRaises(RuntimeError):
                c.validate_archive(path, c.digest(path), dict(expected, sha='c' * 40))
            for kwargs in ({'extra': True}, {'symlink': True}):
                self.archive(path, expected, **kwargs)
                with self.assertRaises(RuntimeError):
                    c.validate_archive(path, c.digest(path), expected)

    def test_stop_polls_until_actual_shutdown_even_if_stop_fails(self):
        with patch.object(c, 'command', side_effect=[RuntimeError('stop failed'),
                          '{"state":"ShuttingDown"}', '{"state":"Shutdown"}']) as run, \
                patch.object(c.time, 'sleep'):
            c.stop_and_confirm('target')
            self.assertEqual(run.call_count, 3)
            self.assertEqual(run.call_args_list[0].args[0], ['gh', 'codespace', 'stop', '--codespace', 'target'])

    def test_unconfirmed_shutdown_fails_with_total_deadline(self):
        with patch.object(c, 'command', return_value='{"state":"Available"}'), \
                patch.object(c.time, 'monotonic', side_effect=[0, 0, 0, 0, 10]), \
                patch.object(c.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Shutdown not confirmed'):
                c.stop_and_confirm('target', timeout=5)


    def test_owned_stop_failure_fails_job_and_retains_diagnostic_report(self):
        proof = c.manifest(IDENT, 'host', DIGEST)
        responses = [SHA, '', json.dumps(self.space()),
                     'RESULT {"ok":true,"result":{"prepared":true}}', '',
                     'RESULT ' + json.dumps({'ok': True, 'result': {'manifest': proof, 'archive_sha256': DIGEST}}), '',
                     'RESULT {"ok":true,"result":{"cleaned":true}}']
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(c.os.environ, RUNNER_TEMP=tmp, GH_TOKEN='test-placeholder',
                           TOKENOMICS_CODESPACE='target', GITHUB_REPOSITORY=c.REPO,
                           GITHUB_SHA=SHA, GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='2'), \
                patch.object(c, 'command', side_effect=responses) as run, \
                patch.object(c, 'digest', return_value=DIGEST), \
                patch.object(c, 'validate_archive'), \
                patch.object(c, 'stop_and_confirm', side_effect=RuntimeError('Actual Shutdown not confirmed')) as stop, \
                patch('builtins.print'):
            self.assertEqual(c.controller(), 1)
            stop.assert_called_once_with('target')
            report = json.loads((Path(tmp) / 'tokenomics-controller/controller-report.json').read_text())
            self.assertTrue(report['claimed'])
            self.assertFalse(report['passed'])
            self.assertFalse(report['shutdown_confirmed'])
            self.assertIn('stop_error', report)
            cp = [call.args[0] for call in run.call_args_list if call.args[0][:3] == ['gh', 'codespace', 'cp']]
            self.assertEqual(len(cp), 2)
            self.assertTrue(cp[0][-1].endswith('/123-2/source.bundle'))
            self.assertTrue(cp[1][-2].endswith('/123-2/evidence.tar.gz'))

    def test_no_stop_if_target_admission_fails(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(c.os.environ, RUNNER_TEMP=tmp, GH_TOKEN='test-placeholder',
                           TOKENOMICS_CODESPACE='target', GITHUB_REPOSITORY=c.REPO,
                           GITHUB_SHA=SHA, GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='2'), \
                patch.object(c, 'command', side_effect=[SHA, '', json.dumps(self.space(state='Available'))]), \
                patch.object(c, 'digest', return_value=DIGEST), \
                patch.object(c, 'stop_and_confirm') as stop, patch('builtins.print'):
            self.assertEqual(c.controller(), 1)
            stop.assert_not_called()
            report = json.loads((Path(tmp) / 'tokenomics-controller/controller-report.json').read_text())
            self.assertFalse(report['claimed'])
            self.assertNotIn('test-placeholder', json.dumps(report))


if __name__ == '__main__':
    unittest.main()
