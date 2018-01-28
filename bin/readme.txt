Dir "Debug" includes bin files and dlls. If the environment is set properly, then you should be able to 
run sph.exe directly.

Exported .obj models are in dir "output". Make sure "output" directory exists in the "Debug"'s parent
directory (that is "bin" here) to successfully export models. 


Demo usage:

During OpenGL real-time rendering, you can press "space" button to pause. Press "space" again will resume simulating.

You can export .obj models of the current frame pressing "o" button. They can be found in ../output. Exporting is 

permitted only when simulation is paused.