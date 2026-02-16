class Trainer:
    def __init__(self, model):
        self.model = model
        gp = self.model
        vae = gp.vae
        dirichlet = vae.dirichlet
        encoder = vae.encoder
        decoder = vae.decoder
        variational_params = [] 
        hyper_params = [] 
        weight_params = [] 