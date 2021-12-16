## Stats 231C - HW 1
## Peter Racioppo

## Problem 3.

e = 2.71828
n = 4
d = n*(n-1)/2 + n + 1
epsilon = 0.1

f_Bound1  <- function(e,m,epsilon){
  Pi_H = m+1
  Exp = 4*exp(-m*epsilon^2/8)
  return(Pi_H*Exp)
}

f_Bound2  <- function(e,m,epsilon){
  Pi_H = (e*m/d)^d
  Exp = 4*exp(-m*epsilon^2/8)
  return(Pi_H*Exp)
}

x1 = 1:13000
Bound1 = x1*0
for(m in x1){
  Bound1[m] = f_Bound1(e,m,epsilon)
}

plot(x1,log(Bound1),type='l',cex.axis=1.5,xlab="",ylab="")
mtext(side=2, line=3, "Log Probability Bound", cex=1.5)
mtext(side=1, line=3, "m", cex=1.5)
lines(x1,x1*log(0+1))
lines(x1,x1*0+log(0.1))
lines(x1,x1*0+log(0.01))
grid()

m_bound1a = which(log(Bound1[5:length(x1)]) < log(1))[1]
m_bound1b = which(log(Bound1[5:length(x1)]) < log(0.1))[1]
m_bound1c = which(log(Bound1[5:length(x1)]) < log(0.01))[1]

m_bound1a # 8,328
m_bound1b # 10,343
m_bound1c # 12,326

# ==================================

x2 = 1:100000
Bound2 = x2*0
for(m in x2){
  Bound2[m] = f_Bound2(e,m,epsilon)
}

plot(x2,log(Bound2),type='l',cex.axis=1.5,xlab="",ylab="")
mtext(side=2, line=3, "Log Probability Bound", cex=1.5)
mtext(side=1, line=3, "m", cex=1.5)
lines(x2,x2*log(0+1))
lines(x2,x2*0+log(0.1))
lines(x2,x2*0+log(0.01))
grid()

m_bound2a = which(log(Bound2[5:length(x2)]) < log(1))[1]
m_bound2b = which(log(Bound2[5:length(x2)]) < log(0.1))[1]
m_bound2c = which(log(Bound2[5:length(x2)]) < log(0.01))[1]

m_bound2a # 89,102
m_bound2b # 91,143
m_bound2c # 93,180

# ==================================
# ==================================

## Problem 4.

H = 1000
N = 600
delta = 0.05
m = 400
f_eL <- function(m,H,delta){
  e_train = sqrt((1/(2*m))*log(2*H/delta))
  e_test = sqrt((1/(2*m))*log(2*1/delta))
  return(c(e_train,e_test))
}

e_train_vec = (0:600)*0
e_test_vec = e_train_vec*0
for(m in 0:600){
  f = f_eL(m,H,delta)
  e_train_vec[m] = f[1]
  e_test_vec[N-m] = f[2]
}

plot(e_train_vec,type='l',cex.axis=1.5,xlab="",ylab="")
lines(e_test_vec,lty=2)
mtext(side=2, line=3, "Error Bar", cex=1.5)
mtext(side=1, line=3, "Training Examples", cex=1.5)
grid()

plot((e_train_vec+e_test_vec)/2,type='l',cex.axis=1.5,xlab="",ylab="")
mtext(side=2, line=3, "Error Bar", cex=1.5)
mtext(side=1, line=3, "Training Examples", cex=1.5)
grid()


