from .conditional_gan_model import ConditionalGAN

def create_model(opt):
	# the legacy test path (TestModel / test.py) was removed; use deblur_fast.py
	model = ConditionalGAN(opt)
	print("model [%s] was created" % (model.name()))
	return model
