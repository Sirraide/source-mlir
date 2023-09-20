import lit.formats

config.name = 'SRCC'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.src']
config.pipefail = False # Turn off nonsense.

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root, 'tests')

config.substitutions.append(('%srcc',
    os.path.join(config.my_src_root, 'srcc')))