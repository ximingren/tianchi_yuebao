with open('./a','r') as f:
    for t in f.readlines():
        if t.split():
            print(t.strip())
