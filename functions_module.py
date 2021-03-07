# import resources
try:
    import numpy as np
    import torch
    from torchvision import models, transforms
    from io import BytesIO
    import requests
    from PIL import  Image
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import uuid
    import  gc
    print("All module loaded in functions_module block ......")

except:
    print("Some Modules are missing.....")
############################################################################################################################
def vgg_model():
    # Instantiate vgg model
    vgg = models.vgg19(pretrained=True).features
    # we dont need VGG parameters so freeze it
    for param in vgg.parameters():
        param.requires_grad_(False)

    return vgg



def image_loader(img_path, max_size=300, shape=None):
    # In case img_path is web address
    if 'http' in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    
    else:
        image = Image.open(img_path).convert('RGB')
    # we dont want image size to be big, because it slow down speed
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    # convert image to tensor and normalize it 
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))
    ])

    image = in_transform(image)[:3,:,:].unsqueeze(0)

    return image

def np_convert(tensor):

    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0,1)

    return image

def get_features(image, model, layers=None):

    
    # Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
    
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
        
    return features

def gram_matrix(tensor):

    _,d,h,w = tensor.size()
    # reshape the tensor so we're multiply the features for every chennels
    tensor = tensor.view(d, h*w)
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram



def style_on_target(model, content_image,style_image):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    content = image_loader(content_image).to(device)

    style = image_loader(style_image, shape=content.shape[-2:]).to(device)

    content_features = get_features(content, model)
    style_features = get_features(style, model)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)

    style_weights = {'conv1_1': 0.95,
                     'conv2_1':0.75,
                     'conv3_1':0.4,
                     'conv4_1':0.4,
                     'conv5_1':0.3}

    content_weight = 1
    style_weight = 1e4
    

    optimizer = optim.Adam([target], lr=0.1)
    steps = 100

    for i in range(1, steps+1):

        target_features = get_features(target, model)

        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0

        for layer in style_weights:

            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape

            style_gram = style_grams[layer]

            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    unique_name = str(uuid.uuid4()) + ".png"
    result_path = "static/uploads/content/" + unique_name
    plt.imsave(result_path, np_convert(target))

    return unique_name

def parameter_gen(content_path, style_path):
    content_path = "static/uploads/content/" + content_path
    style_path = "static/uploads/content/" + style_path

    vgg = vgg_model()

    params = {
        'model':vgg,
        'content_image' : content_path,
        'style_image' : style_path
    }

    f_name = style_on_target(**params)
    gc.collect()
    return f_name

