import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), './saved_model/model_weights.pth')

model2 = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model2.load_state_dict(torch.load('./saved_model/model_weights.pth'))
model2.eval()

torch.save(model2, './saved_model/model.pth')
model3 = torch.load('./saved_model/model.pth')