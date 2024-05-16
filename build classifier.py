import fastai
print(fastai.__version__)

from fastai.vision.all import *
path = Path("/home/krakenmare/Documents/Machine Learning/Resources/Land Classifier Dataset")

Path.BASE_PATH = path


blocks = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),  
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'), 
                 batch_tfms=aug_transforms(mult=2))

dls = blocks.dataloaders(path)
blocks = blocks.new(batch_tfms=aug_transforms(mult=2.0))
dls = blocks.dataloaders(path)
dls.train.show_batch(nrows=3, unique=True)

learn = vision_learner(dls, resnet50, metrics=accuracy)
print(learn.summary())

learn.fine_tune(20)

learn.export('model.pkl')