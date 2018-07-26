from numpy import *

num_place = 10
num_users = 5

ratings = random.randint(11, size=(num_place,num_users))
print(ratings)
did_rate = (ratings != 0)*1
print(did_rate)
nish_ratings = zeroes((num_place,1))
print(nish_ratings)
nish_ratings[0]=2
nish_ratings[5]=5
nish_ratings[9]=3
nish_ratings[1]=4
print(nish_ratings)
ratings = append(nish_ratings,ratings,axis=1)
print(ratings)
did_rate = append(((nish_ratings != 0)*1),did_rate,axis=1)
print(did_rate)

def normalize_ratings(ratings,did_rate):
    num_place = ratings.shape[0]
    ratings_mean = zeros(shape = (num_place,1))
    ratings_norm = zeros(shape = ratings.shape)
        
    for i in range(num_place):
        idx=where(did_rate[i] == 1)[0]
        ratings_mean[i]=mean(ratings[i,idx])
        ratings_norm[i,idx]=ratings[i,idx]-ratings_mean[i]
    return ratings_norm,ratings_mean

 ratings_norm,ratings_mean = normalize_ratings(ratings,did_rate)

num_users = ratings.shape[1]
num_features = 3
place_features = random.randn( num_place, num_features )
user_prefs = random.randn( num_users, num_features )
initial_X_and_theta = r_[place_features.T.flatten(), user_prefs.T.flatten()]
print(place_features)

def unroll_params(X_and_theta, num_users,num_place,num_features):
    first_30 = X_and_theta[:num_place*num_features]
    X = first_30.reshape(( num_features,num_place)).transpose()
    last_18 = X_and_theta[num_place*num_features:]
    theta = last_18.reshape( num_features,num_users ).transpose()
    return X,theta

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_place, num_features, reg_param): 
     X, theta = unroll_params(X_and_theta, num_users,num_place,num_features)
     difference = X.dot(theta.T)*did_rate-ratings
     X_grad = difference.dot(theta)+reg_param*X
     theta_grad = difference.T.dot(X)+reg_param*theta
     return r_[X_grad.T.flatten(),theta_grad.T.flatten()]

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_place, num_features, reg_param):
     X, theta = unroll_params(X_and_theta, num_users,num_place,num_features)
     cost = sum((X.dot(theta.T)*did_rate-ratings)**2)/2
     return cost

from scipy import optimize
reg_param = 30

min_cost_optimize = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta,
                                    args=(ratings, did_rate, num_users, num_place, num_features, reg_param),
                                    maxiter=100,disp=True,full_output=True)


cost, optimal_features_and_user_pref = min_cost_optimize[1],min_cost_optimize[0]
place_features,user_prefs = unroll_params(optimal_features_and_user_pref,num_users,num_place,num_features)
print (place_features)

all_predictions = place_features.dot(user_prefs.T)
print (all_predictions)
predictions_nishita = all_predictions[:, 0:1] + ratings_mean
print (predictions_nishita)