import argparse 
from helpers import loadFile

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("SGDtrace", help="Trace file for SGD solver.")

    parser.add_argument("MBOtrace", help="Trace file for MBO solver.")
    args = parser.parse_args() 


    SGD_trace = loadFile( args.SGDtrace )

    MBO_trace = loadFile( args.MBOtrace )

    SGD_obj = min( SGD_trace['OBJ'] )

    print("SGD obj is ", SGD_obj)

    print("SGD total time is: {:.2f}".format( (SGD_trace['time'][-1] - SGD_trace['time'][0]) / 3600 ) )

    for it, obj in enumerate( SGD_trace['OBJ']):
         if obj <= SGD_obj:
             print( "SGD time to best {:.2f}".format( (SGD_trace['time'][it] - SGD_trace['time'][0]) / 3600 ) )
             break

    print(MBO_trace['OBJ'])
    for it, obj in enumerate( MBO_trace['OBJ'] ):
        if obj < SGD_obj:
            print( "Time to this {:.2f}".format( (MBO_trace['time'][it] - MBO_trace['time'][0]) / 3600 ) )
            break

    print("Total MBO time is {:.2f}".format( (MBO_trace['time'][-1] -  MBO_trace['time'][0]) / 3600 ) )
