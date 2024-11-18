# **Branching Instructions for Group Members**

To ensure smooth collaboration, each group member will work on their own branch. Here’s how to set it up:

---

## **Branch Names**
For example:
- **Melis**: Use the branch name `melis`

---

## **Steps to Create & Use Your Branch**


### **1. Switch to Your Branch**
Run the following command to create and switch to your branch:
```bash
git checkout -b <branch-name>
```

### **2. Make Your Changes**
After making your changes, stage and commit them:
```bash
git add .
git commit -m "Describe your changes here"
```
### **3. Push Your Changes**
Push your changes to the remote repository:
```bash
git push origin <branch-name>
```

### **4. Sync With the Main Branch**
Before merging your work, make sure your branch is up to date with the latest changes from main:
```bash
git checkout main
git pull origin main
git checkout <branch-name>
git merge main
```