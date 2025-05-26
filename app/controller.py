from typing import List

# Create a function to return True/False to find out wether it needs to create vector store again
def DefineControllerList(commandline_arglist:List) -> bool:
    """
    Read in the command line argument and return the flag wether vectoir store creation is neede or not
    
    Args:
        commandline_arglist (List) : list of command line arguments
        
    Returns:
        wether_vector_store_needed (bool) : flag representing wether vector store creation is needed
    """
    # The number of command line argument should be only one
    assert(len(commandline_arglist)) == 2, "The number of arguments in the command line should be only one in addition to the file name"
    
    assert commandline_arglist[1] in ["True", "False"], """The second argument should be either True or False , if vector store creation is needed then True 
    otherwise False
    """
    # extract the flag from command line arguments
    wether_vector_store_needed = commandline_arglist[1]
    # return the flag
    return wether_vector_store_needed
    
    
    
    