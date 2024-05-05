import segmentation_models as sm


def UEfficientNet():
    model = sm.Unet('efficientnetb3',
                    input_shape=(224, 224, 3),
                    classes=1,
                    activation='sigmoid',
                    encoder_weights=None)
    model.custom_name = 'UEfficientNetB3'
    return model
