
x <- 1:10
y <- 11:20
h <- 3

smoother <- function(x,y,h) {
  meanclose <- function(t){ 
    mean(y[abs(x-t) < h])
  }
  sapply(x,meanclose)
}

smoother(x,y,h)