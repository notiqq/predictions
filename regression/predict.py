import core
import sys
import getopt
import pandas as pd

def get_file_paths(argv):
    arg_input = ""
    arg_output = ""
    arg_help = "{0} -i <input> -o <output>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hi:u:o:", ["help", "input=", 
        "user=", "output="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_input = arg
        elif opt in ("-o", "--output"):
            arg_output = arg

    return arg_input, arg_output

def predict(data):
    model = core.NeuralNetwork()
    model.load()
    predictions = model.predict(data)
    return predictions


if __name__ == '__main__':

    input_file, output = get_file_paths(sys.argv)
    df = pd.read_csv(input_file)
    df['prediction'] = predict(df.values)
    df.to_csv(output, index=False)
    
