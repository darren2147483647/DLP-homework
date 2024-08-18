import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.Wq=nn.Linear(dim,dim)
        self.Wk=nn.Linear(dim,dim)
        self.Wv=nn.Linear(dim,dim)
        self.out=nn.Linear(dim,dim)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_image_tokens, _ = x.size()
        q=self.Wq(x)
        k=self.Wk(x)
        v=self.Wv(x)
        q=q.view(batch_size,num_image_tokens,16,768//16)
        k=k.view(batch_size,num_image_tokens,16,768//16)
        v=v.view(batch_size,num_image_tokens,16,768//16)
        q=q.permute(0,2,1,3)
        k=k.permute(0,2,3,1)
        v=v.permute(0,2,1,3)
        score=torch.matmul(q,k)/(768//16)**0.5
        out=torch.matmul(torch.softmax(score,dim=-1),v)
        out=out.permute(0,2,1,3)
        out=out.contiguous().view(batch_size,num_image_tokens,768)
        out=self.out(out)
        return out

        raise Exception('TODO1!')

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    