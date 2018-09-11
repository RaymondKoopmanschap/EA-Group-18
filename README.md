0. Install Intellij
a) For ubuntu 18.04+ 
```
sudo snap install intellij-idea-community --classic --edge
```

b) For ubuntu < 18.04
```
follow instructions at: https://www.jetbrains.com/help/idea/install-and-set-up-product.html
```


1. Open Intellij and craete a random project (this will create the ~/IdeaProjects directory that we need later)


2.  Open terminal and clone this repository into ~/IdeaProjects:
```
cd ~/IdeaProjects
git clone https://github.com/RaymondKoopmanschap/EA-Group-18.git
```
3. In intelliJ open the EA-Group-18 project (File - open and for path input: ~/IdeaProjects/EA-Group-18/) r you should see the EA-Group-18 directory structure

4. Build the project (Ctrl+F9) and rebuild whole project if needed in intelliJ

5. Copy the built submission.jar file to the directory of the unpacked zip downloaded from canvas
```
cp ~/IdeaProjects/EA-Group-18/classes/artifacts/submission_jar/submission.jar ~/Documents/ec/assignmentfiles_2017/
```

6. Run the test:
```
java -jar testrun.jar -submission=player18 -evaluation=BentCigarFunction -seed=3
```
