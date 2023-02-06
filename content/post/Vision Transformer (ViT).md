+++
author = "Rajan Ghimire"
title = "Vision Transformer (ViT)"
date = "2023-02-06"
description = "ViT from scratch in pytorch"
tags = [
    "Computer Vison",
    "PyTorch",

]
+++

<p align="center">
    <img src="https://www.xtrafondos.com/wallpapers/resized/optimus-prime-peleando-4937.jpg?s=large">
</p>

Transformers were widely used in the field of natural language processing when they were first developed.  Many researchers have begun using the Transformer architecture in other domains, like computer vision, as a result of Transformers' success in the field of Natural Language Processing (NLP). One such architecture, called the Vision Transformer, was developed by Google Research and Brain Team to tackle the challenge of image classification.
Naturally, you must have prior knowledge of how Transformers function and the issues it addressed in order to grasp how ViT operates. Before delving into the specifics of the ViT, I'll briefly explain how transformers function.
*If you already understand Transformers, feel free to skip ahead to the next section*

**Vanilla Transformer:** <br>

Previously Recurrent Neural Network (RNN) and LSTM were widely used in Natural Language Processing tasks like next word prediction, machine translation, text generation and more. One of the biggest issues with these RNNs, is that they make use of sequential computation. For example: Suppose, we are translating word “How are you?” to any other language. In order for your code to process the word "you", it has to first go through "are" and then "you". And two other issues with RNNs are: 
- Loss of information: For example, it is harder to keep track of whether the subject is singular or plural as you move further away from the subject.
- Vanishing Gradient: when you back-propagate, the gradients can become really small and as a result, your model will not be learning much

To overcome the problem of RNNs, The Transformer was introduced. Transformers are based on attention and don't require any sequential computation per layer, only a single step is needed. The attention is word-to-word mechanism i.e. the attention mechanism finds how much a word in a sentence is related to all words in the sentence, including the word analyzed with itself. Finally, transformers don't suffer from vanishing gradients problems that are related to the length of the sequences.

**Understanding the Transformer Encoder:**<br>
<p align="center">
    <img width = "500" src="https://quantdare.com/wp-content/uploads/2021/11/transformer_arch.png">
</p>

**Step 1:  Input Embedding**

First layer in Transformer is the embedding layer. This sub-layer converts the input tokens tokenized by the tokenizer into the vectors of dimension 512. Neural networks learn through numbers so each word must be mapped to a vector with continuous values to represent that word.

**Step 2: Positional Encoding**

Position and order of words in a sentence is vital because position of words in sentence defines the grammar and actual semantics of sentence. Recurrent Neural Network take the order of the word into account as they take a sentence word by word in a sequential order. So, we must input some positional information to the embeddings form the first layer as each word in a sentence simultaneously flows through the Transformer encoder / decoder. The model doesn’t have any sense of order/sequence of each word. To incorporate the order of the word, the concept of positional encoding is used. The positional encoding is done using the sine and the cosine function. 

**Step 3: Multi-Headed Attention**

Multi Head attention is the key feature of the transformer. It is the layer that applies mechanism of Self-attention. Attention is a means of selectively weighting different elements in input data, so that they will have an adjusted impact on the hidden states of downstream layers. The attention mechanisms allow a decoder, while it is generating an output word, to focus more on relevant words or hidden states within the network, and focus less on irrelevant information. 
To achieve self-attention, the positional input embedding is fed into 3 distinct fully connected layers to form query(Q), key(k) and value(V) vectors. Here for Example of query is search text on YouTube or google, key is the video title or article title searched for associated with the query text.
Now the query and key undergo dot product multiplication (QKT) to get the score matrix where highest scores are obtained for those words which are to be given more attention in search. Now, scores are scaled down by dividing it by square root of dimensions of queries and keys (√dk). This is done to have more stable gradients, as multiplying values can have exploding gradient problem. Now we have scaled scores. Now, SoftMax is applied to scaled scores to get probability between 0 to 1 for each word, the higher probability words will get more attention and lesser values will be ignored.
Now the matrix after SoftMax is multiplied with value(V) vector. The higher SoftMax will keep the value of word which the model thinks if of higher relevance and Lower scores will be termed as irrelevant. Now the final output matrix is applied to linear layer to perform further processing.<br>
<p align="center">
    <img width = "800" src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-08_at_12.17.05_AM_st5S0XV.png">
</p>
Computing Multi-Head attention: <br>
To make this a multi-headed attention computation the query, key, and value into are split into N vectors before applying self-attention. The split vectors then go through the self-attention process individually. Each self-attention process is called a head. The dimensionality of each head is ‘d_k’ is ( embedding_dim / h) where h is number of heads. Each head produces an output vector that gets concatenated into a single vector before going through the final linear layer.
<p align="center">
    <img width = "300" src="https://blog.scaleway.com/content/images/2019/08/atteq.jpeg">
</p>

**Step 4: The Residual Connections, Layer Normalization, and Feed Forward Network**
The output of multi-head attention is added to original positional input embedding. This is called Residual connection. The idea behind residual connection is learning what’s left of (residual), without learning the new representation. 
The output of residual is applied to Layer Normalization. Here we perform layer normalization in order to prevent the values of output from becoming bigger. We have performed a lot of operations which may cause the values of the layer output to become bigger. So, we use layer normalization to normalize back again. 
The output of layer normalization is applied to a feed forward network. The feed forward network consists of a couple of linear layers with Relu activation in between. Point-wise feed forward is used to further process the attention output and giving it a weighted representation.

**The Vison Transformer:**


<p align="center">
    <img width = "800" src="https://amaarora.github.io/images/ViT.png">
</p>
We are finally prepared to tackle vision transformers now that we have thoroughly explored the internal operation of transformers.<br>

Applying Transformers on images is a challeng for the following reasons: 
- Images convey significantly more information than words, phrases, or paragraphs do, primarily in the form of pixels.
- Even with current hardware, it would be incredibly challenging to focus on every single pixel in the image.
- Instead, using localized focus was a well-liked substitute.
- In fact CNNs do something very similar through convolutions and the receptive field essentially grows bigger as we go deeper into the model's layers, but Tranformers were always going to be computationally more expensive
  

The general architecture can be easily explained in the following five easy steps:

1. Split images into patches.
2. Obtain the Patch Embeddings, also known as the linear embeddings (representation) from each patch.
3. Each of the Patch Embeddings should have position embeddings and a [cls] token.
4. Get the output values for each of the [cls] tokens by passing each one through a Transformer Encoder.
5. To obtain final class predictions, run the representations of [cls] tokens through an MLP Head.


### Step 1 and Step 2:  PatchEmbedding
* * * 
Splitting an image into fixed-size patches and then linearly embedding each one of them using a linear projection layer is one method we use to obtain patch embeddings from an input image.

<p align="center">
    <img width="700" src="https://miro.medium.com/v2/resize:fit:1400/0*kEANaRaJkCPu685t">
    <figcaption align="center" >Picture by paper authors (Alexey Dosovitskiy et al.)</figcaption>  
</p>

However, by employing the 2D Convolution procedure, it is actually possible to combine the two stages into a single step.
If we set the the number of out_channels to 768, and both kernel_size & stride to 16, once we perform the convolution operation (where the 2-D Convolution has kernel size 3 x 16 x 16), we can get the Patch Embeddings matrix of size 196 x 768 like below: [source](https://amaarora.github.io/2021/01/18/ViT.html#the-vision-transformer)
```
# input image `B, C, H, W`
x = torch.randn(1, 3, 224, 224)
# 2D conv
conv = nn.Conv2d(3, 768, 16, 16)
conv(x).reshape(-1, 196).transpose(0,1).shape

>> torch.Size([196, 768])
```
```python
class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = tuple([img_size,img_size])
        patch_size = tuple([patch_size, patch_size])
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

```

### Step 3: CLS TOKEN & Positional Encoding
* * * 
One of the interesting things about the Vision Transformer is that the architecture uses Class Tokens.These Class Tokens are randomly initialized tokens that are prepended to the beginning of your input sequence. The class token has the role of capturing information about the other tokens.<br>

Since the token is randomly initialized, it doesn't have any meaningful data on it by itself. The deeper and more layered the Transformer is,the more information the Class Token can gather from the other tokens in the sequence.<br>

When the Vision Transformer completes the sequence's final classification, it utilizes an MLP head that only considers information from the Class Token of the last layer and no other information. The Class Token appears to be a placeholder data structure that is used to store information that is gleaned from other tokens in the sequence.<br>
**[cls]** token is a vector of size **1 x 768**

<p align="center">
    <img  src="https://miro.medium.com/v2/resize:fit:828/0*F_igiisSnY9tUeAK">
</p>

The positional information of each word within the input sequence is often attempted to be encoded when using transformers to create language models. Each word has a positional encoding that indicates where it should be in the sentence. The Vision Transformer does the same thing by adding a positional encoding to every patch. The top left patch represents the first token, and the bottom right patch represents the last token.
The position embedding is just a tensor of shape $(batchsize,num of patch + 1, embedding size)$ that is added to the projected patches. 

<p align="center">
    <img  src="https://amaarora.github.io/images/vit-03.png">
</p>

so, adding [CLS] token and Positional Encoding to the ```PatchEmbed``` class: 

```python

class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding + cls token + positonal encoding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = tuple([img_size,img_size])
        patch_size = tuple([patch_size, patch_size])
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # [cls] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # positional encoding 
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        # Add CLS token to the patch embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)

        # Adding POS emmbedding 
        x += self.pos_embed
        return x
```

### Step 4: Transformer Encoder
**Attention Block**
```python
class Attention(nn.Module):
    def __init__(self, dim = 768, num_heads=8, qkv_bias=False, qk_scale=None, dropout=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

```
**Multi-Layer Perceptron Block**
```python 
class MLP(nn.Sequential):
    def __init__(self, emb_size: int, L: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, L * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(L * emb_size, emb_size),
        )
```
**Encoder Block**

```python 
class TransformerEncoderBlock(nn.Module):
    def __init__(self,emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,forward_drop_p: float = 0., attn_drp: float = 0.):

        super().__init__()
        self.attention = Attention(dim = emb_size, num_heads=8, qkv_bias=False, qk_scale=None, dropout=attn_drp, proj_drop=0.)
        self.mlp = MLP(emb_size, L=forward_expansion, drop_p=forward_drop_p)

        self.drp_out = nn.Dropout(drop_p) if drop_p > 0. else nn.Identity()
        self.layer_norm_1 =  nn.LayerNorm(emb_size)
        self.layer_norm_2 =  nn.LayerNorm(emb_size)

    
    def forward(self,x):

        x = x + self.drp_out(self.attention(self.layer_norm_1(x)))
        x = x + self.drp_out(self.mlp(self.layer_norm_2(x)))

        return x

```
**Wrapping all:**
```python 
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

```

### Step 5: The classification Head and VIT:

```python 
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
```