import torch.nn as nn

from Transformer.layers.ScaleDotProductAttention import ScaleDotProductAttention

class MultiheadAttention(nn.Module):

    def __init__(self, d_model,n_head):
        super().__init__()
        self.attention=ScaleDotProductAttention()
        self.n_head=n_head
        #가중치 행렬들
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_concat=nn.Linear(d_model,d_model)

        #왜 입출력 차원이 동일하지?
        #논문에서는 d_model/n_head였는데.. 게다가 head층마다 W가중치가 각각 다르다고 했었는데

    def forward(self,q,k,v,mask=None):
        
        #가중치 W와 곱해지기 전 q,k,v는 완전히 동일한 재료 X
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)
        #가중치와 곱한 후 head로 분할
        q,k,v=self.split(q), self.split(k), self.split(v)
        #제가 알기론 attention에 ScaleDotProductAttention객체를 생성했으니 그 안에 forward함수를
        #사용하려면 attention.forward()하고 호출을 해야하는데
        #pytorch의 nn.Module을 상속받으면 forward는 마치 예약어처럼 되어있어서 call 내장함수에 의해
        #입력하지않아도 ()만 붙여주면 자동으로 호출이 된다
        #그리고 이렇게 해야 역전파가 가능하도록 pytorch가 추척한다! 절대 직접 호출하지 말것 중요
        out, attention=self.attention(q,k,v,mask=mask)
        #out 결과는 V행렬 다시 차원 합치고 w_concat 가중치 곱하기
        out=self.concat(out)
        out=self.w_concat(out)

        return out

    
        #여기서 split이 된다!
        #효율적인 연산을 위해서 먼저 가중치 내적 한 다음에 곱
    def split(self,tensor):
        batch_size,length,d_model=tensor.size()
        d_tensor=d_model//self.n_head
        print(tensor.shape)
        tensor=tensor.view(batch_size,length,self.n_head,d_tensor).transpose(1,2)
        print(tensor.shape)
        return tensor
    
    def concat(self,tensor):
        batch_size,n_head,length,d_tensor=tensor.size()
        print(tensor.shape)
        tensor=tensor.view(batch_size,length,d_tensor*n_head)
        print(tensor.shape)
        return tensor



