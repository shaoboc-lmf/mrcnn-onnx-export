import torch
import io
import onnxruntime

def onnx_print_io_meta(ort_session):

    print("\n==ort inputs==")
    for ort_inputs_meta in ort_session.get_inputs():
        # print(dir(ort_inputs_meta))
        print(ort_inputs_meta.name)
        print(ort_inputs_meta.shape)
        print(ort_inputs_meta.type)
        print()

    print("\n==ort outputs==")
    for ort_outputs_meta in ort_session.get_outputs():
        # print(dir(ort_outputs_meta))
        print(ort_outputs_meta.name)
        print(ort_outputs_meta.shape)
        print(ort_outputs_meta.type)
        print()

class ClipMod(torch.nn.Module):

    def __init__(self):

        super(ClipMod, self).__init__()

    def forward(self, box, upper_bound):
        print("forward, upper:", upper_bound)
        return box.clamp(max=upper_bound)
        # return torch.min(box, upper_bound)


@torch.no_grad()
def test():

    clip_mod = ClipMod()
    clip_mod.eval()
    
    i = torch.tensor([0.1, 1.0, 1.1, 1.2])
    u = torch.tensor(1.)
    o = clip_mod(i, u)

    print("==torch==")
    print("i", i)
    print("o", o)

    onnx_io = io.BytesIO()
    torch.onnx.export(clip_mod,
                      (i, u),
                      onnx_io,
                      do_constant_folding = True,
                      opset_version = 11,
                      verbose = True,
                      input_names  = ['input', 'upper_bound'],
                      output_names = ['output'])

    ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
    onnx_print_io_meta(ort_session)

    i_ort = i.numpy()
    u_ort = o.numpy()

    print("==onnxruntime upper_bound 1.==")    
    o_ort = ort_session.run(None, {
        "input"       : i_ort,
        "upper_bound" : u_ort
    })
    print(o_ort)
    
    # o_ort = ort_session.run(None, {
    #     "input"       : i_ort,
    # })
    # print("==onnxruntime==")
    # print(o_ort)

    print("==onnxruntime upper_bound 1.1==")
    u_ort = torch.tensor(1.1).numpy()
    o_ort = ort_session.run(None, {
        "input"       : i_ort,
        "upper_bound" : u_ort
    })
    print(o_ort)

if __name__ == "__main__":
    test()
    
