from mmengine import Registry

FILTERDATASET = Registry('dataset',
                         scope='Net',
                         locations=['Net.dataset.track_dataset'])
PARAMS = Registry('params', scope='Net', locations=['Net.params'])

MODELS = Registry('model', scope='Net', locations=['Net.models'])
