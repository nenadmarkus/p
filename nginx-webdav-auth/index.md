---
title: Configuring Nginx WebDAV with custom authentication
date: 2025-03-30
excerpt: A minimalistic implementation of Gaussian process regression in Python
tags: nginx, webdav, authentication
layout: default
katex: false
---

Requirements:

* serve WebDAV over Nginx
* use custom authentication based on requested path
* [basic authentication header](https://en.wikipedia.org/wiki/Basic_access_authentication) (`Authorization: Basic <credentials>`)
* dockerize the solution

I lost about two days to set this up despite using ChatGPT, DeepSeek, Grok and Gemini for help.
So I suppose the contents of this post might be useful to some people.

### Nginx config

The first line -- loading `dav_ext` module -- is crucial.
It caused a lot of pain and none of the AI chatbots were helpful.
I finally got the hint how to solve it at the following link:
<https://nginx-extras.getpagespeed.com/modules/dav-ext/>.

```nginx
load_module modules/ngx_http_dav_ext_module.so;

events {
    worker_connections 1024;
}

http {
    # limit maximum file upload size to 2MB
    client_max_body_size 2000M;

    server {
        listen 80;

        # WebDAV location
        location / {
            auth_request /auth;
            root /mnt;
            dav_methods PUT DELETE MKCOL COPY MOVE;
            dav_ext_methods PROPFIND OPTIONS;
            create_full_put_path on;
            dav_access user:rw group:rw all:rw;
            auth_request_set $auth_uri $uri;
        }

        # internal auth subrequest
        location = /auth {
            internal;
            proxy_pass http://127.0.0.1:8080/auth?path=$request_uri&method=$request_method;
            proxy_pass_request_body off;
            proxy_set_header Content-Length "";
            proxy_set_header Authorization $http_authorization;
        }

        # error handling for unauthorized access
        error_page 401 = @unauthorized;
        location @unauthorized {
            return 401 "Unauthorized\n";
        }
    }
}
```

### Auth in Golang

This is set up so that `username`, when provided with proper password in basic auth scheme, has access to folder `username`.
Of course, you can modify the code below for a different set up, but the relevant data is passed to the `/auth` endpoint: path, method and credentials.

```golang
package main

import (
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// path: token
var accessList = map[string]string{
	"user1":    "pass1",
	"user2":    "pass2",
	"testuser": "testpass",
}

func main() {
	mkDirs("/mnt")

	http.HandleFunc("/auth", authHandler)

	fmt.Println("* starting auth server")
	http.ListenAndServe(":8080", nil)
}

func mkDirs(root string) {
	for path, _ := range accessList {
		err := os.MkdirAll(filepath.Join(root, path), os.ModePerm)
		if err != nil {
			panic(err)
		}
		if err := os.Chmod(filepath.Join(root, path), 0777); err != nil {
			panic(err)
		}
	}
}

func authHandler(w http.ResponseWriter, r *http.Request) {

	username := ""
	password := ""
	ok := false
	authHeader := r.Header.Get("Authorization")
	if strings.HasPrefix(authHeader, "Basic ") {
		encoded := authHeader[len("Basic "):]
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err == nil {
			parts := strings.SplitN(string(decoded), ":", 2)
			if len(parts) == 2 {
				username, password = parts[0], parts[1]
				ok = true
			}
		}
	}

	path := r.URL.Query().Get("path")
	method := r.URL.Query().Get("method")

	fmt.Printf("* auth request (%d): "+username+":"+password+" -> "+method+" -> "+path+"\n", ok)

	if !ok || !isValidUser(username, password, path) {
		w.Header().Set("WWW-Authenticate", `Basic realm="Restricted"`)
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		fmt.Println("* auth not ok")
		return
	}

	fmt.Println("* auth ok")
	w.WriteHeader(http.StatusOK)
}

func isValidUser(username, password, path string) bool {

	expectedPassword, exists := accessList[username]
	if !exists {
		fmt.Printf("* `%s` does not exist\n", username)
		return false
	}

	// cerify password
	if subtle.ConstantTimeCompare([]byte(password), []byte(expectedPassword)) != 1 {
		fmt.Println("* wrong password")
		return false
	}

	// check if user has access to the requested path
	if !strings.HasPrefix(path, fmt.Sprintf("/%s/", username)) && strings.Compare(path, fmt.Sprintf("/%s", username)) != 0 {
		fmt.Printf("* invalid access path")
		return false
	}

	// we're done
	return true
}
```

### Dockerfile

Use `docker build -t nginxwebdavauth .` and then `docker run -it --rm -v "$(pwd)/mnt:/mnt" -p 8080:80 nginxwebdavauth`.

```dockerfile
# docker build -t nginxwebdavauth .
# docker run -it --rm -v "$(pwd)/mnt:/mnt" -p 8080:80 nginxwebdavauth

# STAGE #1: Build the Go auth binary
FROM golang:1.21 AS go-builder
WORKDIR /app
COPY auth.go .
RUN go mod init auth
RUN CGO_ENABLED=0 GOOS=linux go build -o auth auth.go

# STAGE #2: Set up Nginx and include the Go binary
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y nginx-full nginx-extras

WORKDIR /app

COPY --from=go-builder /app/auth /app/auth
COPY nginx.conf /etc/nginx/nginx.conf

CMD ["/bin/sh", "-c", "nginx -g 'daemon on;' & /app/auth"]
```