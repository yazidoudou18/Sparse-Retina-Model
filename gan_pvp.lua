local pvDir = os.getenv("HOME") .. "/OpenPV";
package.path = package.path .. ";" .. pvDir .. "/parameterWrapper/?.lua"; 
local pv = require "PVModule";
local pvpFileLocation = "Ganglion_pvp_file";
dofile("Retina_CIFAR.lua");
os.execute("mkdir -p " .. pvpFileLocation);
local file = io.open("/home/twatkins/Workspace/Retina/input/mixed_cifar.txt");
for i = 1,50000 do
   local line = file:read();
   print(line);
   pvParams.Image.inputPath = line;
   local file_name = "temp";
   local params = io.open(file_name .. ".params","w");
   io.output(params);
   pv.printConsole(pvParams);
   io.close(params);
   os.execute("mpirun -np 1 " .. pvDir .. "/../build/tests/BasicSystemTest/Release/BasicSystemTest -p " .. file_name .. ".params -t 4");
   local fname = string.sub(line, -19, -19) .. "_"
              .. string.sub(line, -17, -17) .. "_"
              .. string.sub(line, -15, -5)  .. ".pvp";
   print(fname);
   os.execute("mv /home/twatkins/Workspace/Retina/output/Retina_CIFAR10/GanglionON.pvp " .. pvpFileLocation .. "/GanON" .. fname );
   os.execute("mv /home/twatkins/Workspace/Retina/output/Retina_CIFAR10/GanglionOFF.pvp " .. pvpFileLocation .. "/GanOFF" .. fname );
end
io.close(file);
   
