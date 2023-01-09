from torch import nn


def count_params(model):
    if type(model) == nn.DataParallel:
        return model.module.count_params()
    return model.count_params()


def save_state_dict(model):
    if type(model) == nn.DataParallel:
        return model.module.state_dict()
    return model.state_dict()


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def load_conformer_pretrained(
    no_hidden_layers: int = 2,
    pretrained_name: str = "facebook/wav2vec2-conformer-rel-pos-large",
):
    # load pretrained big
    wav2vec2_model = Wav2Vec2ConformerForPreTraining.from_pretrained(pretrained_name)
    wav2vec2_conformer = wav2vec2_model.wav2vec2_conformer.encoder
    wav2vec2_conformer.layers = nn.ModuleList(
        [wav2vec2_conformer.layers[i] for i in range(no_hidden_layers)]
    )
    return wav2vec2_conformer
