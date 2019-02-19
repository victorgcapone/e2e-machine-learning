import yaml

with open("configs.yaml", "r") as conf:
    config = yaml.load(conf)


if __name__ == '__main__':
    print(config)
