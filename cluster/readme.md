### Note: you have to start your VPN before trying out any of the following commands!

# Simple Job Example
Note: You can run a simple job without the the whole ssh configuration described below.
### Send job to the cluster
```
kubectl apply -f remote-job.yml
```
### Check your pod's identifier
```
kubectl get pods
```
### Check the logs (in our case, the printed statement)
```
kubectl logs dsw-job-[identifier]
```
### Clear the job
```
kubectl delete job dsw-job
```

# SSH Configuration
Note: To send PVCs and Deployments to the cluster, you will need an ssh key.
### Generate SSH Key (with a passphrase please)
```
ssh-keygen
```
### Upload your key to K8s as a secret
```
kubectl create secret generic dsw-secret --from-file=authorized_keys=~/.ssh/id_[rsa].pub
```
### Check that the secret was created
```
kubectl get secrets
```
### Create ssh config file
```
nano ~/.ssh/config
```
Or right-click, create new text file, then delete the .txt extension.
### Add the following block
```
Host kubernetes
    HostName localhost
    Port 44414
    User root
    IdentityFile ~/.ssh/id_[rsa]
```

# Docker
Start your Docker Desktop first and foremost.

### Build your image (takes a couple of minutes)
```
docker buildx build --platform linux/amd64 -t jaikushwaha/dsw-python:3.11.9 .
```
### Start a container from your image
```
docker run jaikushwaha/dsw-python:3.11.9
```
### Push your image to Dockerhub (also takes a couple of minutes)
```
docker push jaikushwaha/dsw-python:3.11.9
```

# PVC
### Create PVC
```
kubectl apply -f storage.yml
```
### Check that it's been created
```
kubectl get pvc
kubectl describe pvc dsw-pvc
```

# Deployment
### Launch your deployment onto the cluster
```
kubectl apply -f remote-deployment.yml
```
### Check that it's been applied successfully (sometimes it takes 1-2 minutes)
```
kubectl get deployments
kubectl get pods
kubectl describe pod [pod-name]
```

# Port-Forwarding
Create the tunnel between your machine and the cluster
```
kubectl port-forward [pod-name] 44414:22
```

# Always shut down your deployment when you're done!
This sets the replicas to 0 and doesn't delete the deployment. You can rescale it later with `replicas=1`.
```
kubectl scale deployment dsw --replicas=0
```
Alternatively, this deletes the deployment and you need to reapply it later:
```
kubectl delete deployment dsw
```

# Bonus
### Screen cheat sheet
<<<<<<< HEAD
https://gist.github.com/jctosta/af918e1618682638aa82
=======
https://gist.github.com/jctosta/af918e1618682638aa82
>>>>>>> 9f1190e4675a195094103865ab4361bbcda21cbe
