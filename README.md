0. Install Intellij
```
sudo snap install intellij-idea-community --classic --edge
```
1. clone this repository into ~/IdeaProjects:
```
cd ~/IdeaProjects
git clone https://github.com/RaymondKoopmanschap/EA-Group-18.git
```
2. Open intelliJ, in left bar you should see the EA-Group-18 directory structure

3. Build the project in intelliJ (Ctrl + F9)

4. Copy the built submission.jar file to the directory of the unpacked zip downloaded from canvas
```
cp ~/IdeaProjects/EA-Group-18/classes/artifacts/submission_jar/submission.jar ~/Documents/ec/assignmentfiles_2017/
```

5. Run the test:
```
java -jar testrun.jar -submission=player18 -evaluation=BentCigarFunction -seed=3
```
