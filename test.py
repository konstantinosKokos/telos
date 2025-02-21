import torch
from src.telos.deduction import Judgement, Trace, model
from src.telos.syntax import Variable, Next, Disjunction, Implies, eventually
from src.telos.algebras.goedel import Goedel

x, y, z, w = map(Variable, 'xyzw')
batch_size = 1
trace_size = 2

trace = Trace({
    x: torch.rand(batch_size, trace_size, requires_grad=True),
    y: torch.rand(batch_size, trace_size, requires_grad=True),
    z: torch.rand(batch_size, trace_size, requires_grad=True),
    w: torch.rand(batch_size, trace_size, requires_grad=True),
})

conclusion = eventually(Implies(x, Next(Disjunction(y, z))))
judgement = Judgement(trace, conclusion)
algebra = Goedel()

output = model(algebra)(judgement)
loss = torch.nn.functional.binary_cross_entropy(output, torch.ones(batch_size))
print(loss)
#loss.backward()