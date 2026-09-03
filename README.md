Run "py main.py" for the calculator that only uses floats and int\
This is fast, but can quickly lose precision because of floating point errors\
Just follow the prompts to input the system of equations and you should get an output if you did everything correctly\
If you get an error, read it. If it says that the augmented matrix was invalid, something was wrong with your input\
If it's another error, then it's a formatting or computation error, and the means there's an unseen bug in the code

Run "py fractional_version.py" for the calculator that can use floats, ints, and fractions\
The program will prompt you if you want to use floats and ints, or fractions\
Fractions are infinitely more precise than floats, but make the calculations much slower\
Just follow the prompts to input the system of equations and you should get an output if you did everything correctly\
If you get an error, read it. If it says that the augmented matrix was invalid, something was wrong with your input\
If it's another error, then it's a formatting or computation error, and the means there's an unseen bug in the code

When using one of the 2 prior commands, you can type in the values for the matrix manually, or you can use a .csv file\
You can look at the example .csv files that are given to see how they should be formatted



Run "py timing_test.py" to test the 2 functions that are available and see how long they take to run\
The parameters for timing are explained with comments are in the code and are pretty self explanatory

I've personally found that the column by column method I made is faster when using the Fraction class, and that the Gauss Jordan elimination algorithm is faster with floats. \
Feel free to switch to whichever algorithm you want to use/whichever will work faster on your machine. \
Here's the data I used to come to this conclusion: https://docs.google.com/spreadsheets/d/1qa8tAm3_GasE6jLSZKdni2R_my4WJndrSQPZkayF0sc/edit?usp=sharing