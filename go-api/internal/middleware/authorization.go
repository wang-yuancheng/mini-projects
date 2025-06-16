package middleware

import (
	"errors"
	"net/http"

	log "github.com/sirupsen/logrus"
	"github.com/wang-yuancheng/go-api/api"
	"github.com/wang-yuancheng/go-api/internal/tools"
	"github.com/wang-yuancheng/mini-projects/go-api/api"
)

var UnAuthorizedError = errors.New("Invalid username or token.")

func Authorization(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r*http.Request){
		var username string = r.URL.Query().Get("username")
		var token = r.Header.Get("Authorization")
		var err error
		
		if username == "" || token == "" {
			log.Error(UnAuthorizedError)
			api.RequestErrorHandler(w, UnAuthorizedError)
			return
		}

		var database *tools.DatabaseInterface
		database, err = tools.NewDatabase()
		if err != nil {
			api.InternalErrorHandler(w)
			return
		}

		var loginDetails *tools.loginDetails
		loginDetails = (*database).GetUserLoginDetails(username)

		if (loginDetails == nil || (token != (*loginDetails).AuthToken)){
			log.Error(UnAuthorizedError)
			api.RequestErrorHandler(w, UnAuthorizedError)
			return
		}
		next.ServeHTTP(w, r)
	})
}