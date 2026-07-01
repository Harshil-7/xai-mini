import pickle

with open("models/rgcn_model.pt.mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

print(type(mappings))

if isinstance(mappings, dict):
    print("\nKeys:")
    print(mappings.keys())

    for k, v in mappings.items():
        print("\n==========")
        print("KEY:", k)
        print("TYPE:", type(v))

        try:
            print("Length:", len(v))
        except:
            pass

        if isinstance(v, dict):
            print("First 5 items:")
            for i, (kk, vv) in enumerate(v.items()):
                print(kk, "->", vv)
                if i == 4:
                    break