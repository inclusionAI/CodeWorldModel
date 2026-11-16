"""Dependency checks preserve installed identities and reject unreviewed conflicts."""
import importlib.util
from pathlib import Path
import sys

import pytest
from packaging.tags import Tag


@pytest.fixture
def checker(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/check_omni_installed_requirements.py'
    spec = importlib.util.spec_from_file_location('h3_requirement_checker_test', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def distribution(checker, name, version='1.0', requirements=(), wheel_tags=()):
    return checker.InstalledDistribution(name=name, version=version,
        requirements=tuple(requirements), wheel_tags=tuple(wheel_tags))


def test_missing_and_version_conflicts_keep_original_requirement(checker):
    environment = [distribution(checker, 'consumer', requirements=['Missing_Pkg>=2', 'provider>=2']),
                   distribution(checker, 'provider', '1.5')]
    conflicts = checker.find_conflicts(environment)
    assert [(c.distribution, c.requirement, c.installed_version) for c in conflicts] == [
        ('consumer', 'Missing_Pkg>=2', None), ('consumer', 'provider>=2', '1.5')]
    assert checker.unexpected_conflicts(conflicts) == conflicts


def test_first_sys_path_identity_and_unselected_extras_match_pip(checker):
    environment = [distribution(checker, 'consumer', requirements=[
        'provider>=2', 'optional-provider; extra == "audio"']),
        distribution(checker, 'provider', '2.1'), distribution(checker, 'provider', '1.0')]
    assert checker.find_conflicts(environment) == []
    environment[1], environment[2] = environment[2], environment[1]
    conflicts = checker.find_conflicts(environment)
    assert [(c.required_name, c.installed_version) for c in conflicts] == [('provider', '1.0')]


def test_reviewed_pair_does_not_exempt_other_dependents_or_missing_packages(checker):
    environment = [distribution(checker, 'qwen-tts', requirements=['transformers<5', 'accelerate>=1']),
        distribution(checker, 'unreviewed', requirements=['transformers<5']),
        distribution(checker, 'transformers', '5.0')]
    conflicts = checker.find_conflicts(environment)
    unexpected = checker.unexpected_conflicts(conflicts)
    assert [(c.distribution, c.required_name) for c in unexpected] == [
        ('qwen-tts', 'accelerate'), ('unreviewed', 'transformers')]
    assert len(conflicts) == 3


def test_platform_exception_does_not_exempt_other_wheels(checker):
    supported = frozenset({Tag('cp312', 'cp312', 'manylinux_2_28_x86_64')})
    environment = [distribution(checker, name, wheel_tags=('cp36-cp36m-manylinux1_x86_64',))
                   for name in ('decord', 'unreviewed')]
    conflicts = checker.find_platform_conflicts(environment, supported)
    assert [c.distribution for c in conflicts] == ['decord', 'unreviewed']
    assert [c.distribution for c in checker.unexpected_platform_conflicts(conflicts)] == ['unreviewed']


def test_installed_metadata_is_read_without_rewriting_identity(checker, monkeypatch):
    class Metadata:
        metadata = {'Name': 'Some_Package'}
        version = '2.0+local'
        requires = ['Dependency_Name>=2']

        def read_text(self, name):
            assert name == 'WHEEL'
            return 'Wheel-Version: 1.0\nTag: py3-none-any\n'

    record = Metadata()
    monkeypatch.setattr(checker, 'distributions', lambda: [record])
    assert checker.installed_distributions() == [distribution(checker, 'some-package', '2.0+local',
        requirements=['Dependency_Name>=2'], wheel_tags=['py3-none-any'])]
    assert record.metadata == {'Name': 'Some_Package'} and record.requires == ['Dependency_Name>=2']


@pytest.mark.parametrize('unexpected, expected_status', [(False, 0), (True, 1)])
def test_command_status_distinguishes_reviewed_and_unexpected(checker, monkeypatch, capsys,
                                                            unexpected, expected_status):
    environment = [distribution(checker, 'qwen-tts', requirements=['transformers<5']),
                   distribution(checker, 'transformers', '5.0')]
    if unexpected:
        environment.append(distribution(checker, 'consumer', requirements=['cache-dit==1.3.0']))
    monkeypatch.setattr(checker, 'installed_distributions', lambda: environment)
    assert checker.main() == expected_status
    output = capsys.readouterr()
    assert 'reviewed: qwen-tts' in output.out
    assert ('unexpected: consumer' in output.err) == unexpected
