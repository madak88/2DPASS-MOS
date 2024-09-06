from ruamel.yaml import YAML

path = "config/label_mapping/semantic-kitti-mos.yaml"

yaml = YAML()
yaml.preserve_quotes = True

with open(path, 'r') as file:
    data = yaml.load(file)

data['split']['valid'], data['split']['test'] = data['split']['test'], data['split']['valid']

with open(path, 'w') as file:
    yaml.dump(data, file)
