from .u_net import *

def update_parameters(model, loss, step_size=0.5, first_order=False):
    """Update the parameters of the model, with one step of gradient descent.
    Parameters
    ----------
    model : `MetaModule` instance
        Model.
    loss : `torch.FloatTensor` instance
        Loss function on which the gradient are computed for the descent step.
    step_size : float (default: `0.5`)
        Step-size of the gradient descent step.
    first_order : bool (default: `False`)
        If `True`, use the first-order approximation of MAML.
    Returns
    -------
    params : OrderedDict
        Dictionary containing the parameters after one step of adaptation.
    """
    grads = torch.autograd.grad(loss, model.meta_parameters(),
        create_graph=not first_order)

    params = OrderedDict()
    for (name, param), grad in zip(model.meta_named_parameters(), grads):
        params[name] = param - step_size * grad

    return params