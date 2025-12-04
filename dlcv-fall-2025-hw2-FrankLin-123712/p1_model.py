''' 
Our conditional ddpm model is adpated from : 
[Conditional_Diffusion_MNIST/script.py](https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py)
'''
import os, sys, logging
import torch
import torch.nn as nn
import numpy as np

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        # before one hot encoding c.shape = (n_sample,)
        # after one hot encoding c.shape = (n_sample, n_classes)
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]  # context_mask.shape = (n_sample, 1)
        context_mask = context_mask.repeat(1,self.n_classes) # context_mask.shape = (n_sample, n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask # c.shape = (n_sample, n_classes)
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

def ddpm_schedules(b1, b2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert b1 < b2 < 1.0, "b1 and b2 must be in (0, 1)"

    b = (b2 - b1) * torch.arange(0, T+1, dtype=torch.float32) / T + b1
    sqrtb = torch.sqrt(b)
    a = 1 - b
    log_a = torch.log(a)
    ab = torch.cumsum(log_a, dim=0).exp()

    sqrtab = torch.sqrt(ab)
    oneover_sqrta = 1 / torch.sqrt(a)

    sqrtmab = torch.sqrt(1 - ab)
    ma_over_sqrtmab = (1 - a) / sqrtmab

    return {
        "sqrtb":                      sqrtb,  # \sqrt{\beta_t}
        "a":                              a,  # \alpha_t
        "ab":                            ab,  # \bar{\alpha_t}
        "oneover_sqrta":      oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrtab":                    sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab":                  sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "ma_over_sqrtmab":  ma_over_sqrtmab,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model

        logging.info(f"ContextUnet have been moved to {device}")

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

        logging.info("DDPM model have been initialized successfully.")

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T) with shape (n_sample,)
        noise = torch.randn_like(x)  # eps ~ N(0, 1) with shape (n_sample, 3, 28, 28)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  
        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0):
        """
        Classifier-free diffusion guidance sampling with guidance
        """
        
        x_i = torch.randn(n_sample, *size).to(device) # shape = (n_sample, 3, 28, 28)
        c_i = torch.arange(0,10).to(device) # shape = (10,)
        c_i = c_i.repeat(int(n_sample/c_i.shape[0])) # shape = (n_sample,)

        c_mask = torch.zeros_like(c_i).to(device) # shape = (n_sample,)

        c_i = c_i.repeat(2)       # shape = (n_sample*2, )
        c_mask = c_mask.repeat(2) # shape = (n_sample*2, )
        c_mask[n_sample:] = 1 # second half of batch predict eps without context
        
        logging.debug(f"The shape of x_i : {x_i.shape}")
        logging.debug(f"The shape of c_i : {c_i.shape}")
        logging.debug(f"The shape of c_mask : {c_mask.shape}")

        x_i_buf = []  # storing the intermediate version of image

        for i in range(self.n_T, 0, -1):
            # logging.info(f'sampling timestep {i}')
            ts = torch.tensor([i / self.n_T]).to(device) # shape (1,)
            ts = ts.repeat(n_sample, 1, 1, 1) # shape = (n_sample, 1, 1, 1)

            x_i = x_i.repeat(2, 1, 1, 1) # shape = (n_sample*2, 3, 28, 28)
            ts = ts.repeat(2, 1, 1, 1)   # shape = (n_sample*2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0 # shape (n_sample, 3, 28, 28)

            eps_pred = self.nn_model(x_i, c_i, ts, c_mask) # shape (n_sample*2, 3, 28, 28)
            eps_pred_c = eps_pred[:n_sample] # first half of eps predicted with context
            eps_pred_nc = eps_pred[n_sample:] # second half of eps predicted without context
            eps = (1+guide_w)*eps_pred_c - guide_w*eps_pred_nc

            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - self.ma_over_sqrtmab[i] * eps) + self.sqrtb[i] * z

            if i%200==0 or i==self.n_T:
                x_i_buf.append(x_i.detach().cpu().numpy())

        x_i_buf = np.array(x_i_buf)

        return x_i, x_i_buf

if __name__ == "__main__":

    LOG_FILE = './log/p1/p1_model.log'
    LOG_LEVEL = logging.DEBUG
    os.makedirs("./log/p1", exist_ok=True)

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Construct DDPM model.")

    model = ContextUnet(3).to("cuda")
    ddpm = DDPM(nn_model=model, betas=[1e-4, 2e-2], n_T=1000, device="cuda")

    logging.info("ddpm architecture : ")
    logging.info(f'{ddpm}')
    
