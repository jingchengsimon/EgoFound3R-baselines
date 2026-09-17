"""Current DSW allocation must not silently connect an old run to a new port."""
import json

from formal_evaluation import taskctl


def test_current_instance_overlays_connection_without_persisting(tmp_path):
    key = tmp_path / 'id_ed25519'
    key.write_text('fixture')
    key.chmod(0o600)
    path = tmp_path / 'instance.json'
    path.write_text(json.dumps({
        'instance_id': 'Nv48g9123', 'host': 'dsw.example.test', 'port': 9123,
        'key': str(key), 'entrance': '/mnt/cpfs',
    }))
    instance = taskctl.load_instance_config(path)
    registry = {'run_template': {'ssh': {'host': 'old.example.test', 'key': '/old/key'}},
                'policy': {'resource_nodes': [5000], 'resource_node_entrances': {'5000': '/mnt/workspace'}}}
    taskctl.apply_instance_config(registry, instance)
    assert registry['run_template']['ssh'] == {'host': 'dsw.example.test', 'key': str(key)}
    assert registry['policy']['resource_nodes'] == [9123]
    assert registry['policy']['resource_node_entrances'] == {'9123': '/mnt/cpfs'}
    assert registry['run_template']['_current_instance']['instance_id'] == 'Nv48g9123'


def test_old_port_is_rejected_before_ssh(tmp_path):
    run = {'_current_instance': {'port': 9123}, 'ssh': {'host': 'dsw.example.test', 'key': str(tmp_path)}}
    try:
        taskctl._ssh(run, 5000, 'true')
    except taskctl.TaskError as error:
        assert 'INSTANCE_PORT_MISMATCH' in str(error)
    else:
        raise AssertionError('stale node port was accepted')
