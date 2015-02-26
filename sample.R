
x <- 1:40
y <- 41:80
h <- 7

smoother <- function(x,y,h) {
  meanclose <- function(t) 
    mean(y[abs(x-t) < h])
  sapply(x,meanclose)
}

smoother(x,y,h)