
if __name__ == "__main__":   
    itensor = np.arange(25).reshape(5,5)[None][None]
    conv2d = Conv2d(in_feature=1,out_feature=1,kernel_size=3,padding = 0)
    # forward
    print(conv2d(itensor))
    print("ddd")
    # backward
    G = np.array([[[[11., 13., 15.],
         [21., 23., 25.],
         [31., 33., 35.]]]])
    print(conv2d.backward(G))
