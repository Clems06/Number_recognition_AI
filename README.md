# Number_recognition_AI

This is a Supervised Neural Network that learns how to recognise digits using backpropagation. It needs the MNIST database to work, and I couldn't upload it due to its big size. If you want to use the program, please install the csv files of the database.

The problem with this process is that it is very long, and you wouldn't want to wait each time you execute it. That is why I made it so if you press the "g" key, it saves the net in the *save_net.json* as a JSON object. The file *number_recognition.py* is used to create a NEW net. Once you have a net saved, use *continue_process.py* to continue without losing all the progress.

*draw_and_recognise.py* opens Tk in an interface where you can draw the digit and it recognises it. This has flaws though, as you need to use the same size and alignment as the MNIST database, and I couldn't quite make the same line thickness.

# REQUIREMENTS:
This program needs the non built-in libraries *keyboard* and *jsonpickel*.
