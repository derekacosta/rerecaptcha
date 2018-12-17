import Clarifai from 'clarifai';
import * as fs from 'fs';
var router = require('express').Router();
var React = require('react');
var ReactDOMServer = require('react-dom/server');
var ReactRouter = require('react-router');
var Redux = require('redux');
var Provider = require('react-redux').Provider;
var path = require('path')

function reducer(state) {
    return state;
}

let image = 'https://picsum.photos/500/500?image=' + Math.floor(Math.random() * Math.floor(1000)).toString();

const app = new Clarifai.App({apiKey: 'd18fb427d30445f58b2e86da02d29303'});

function getFacts(image) {
    return new Promise((resolve, reject) => {

        resolve(app.models.initModel({id: Clarifai.GENERAL_MODEL, version: "aa7f35c01e0642fda5cf400f543e7c40"}).then(generalModel => {
            return generalModel.predict(image);
        }).then(response => {
            return response['outputs'][0]['data']['concepts'].map(x => x.name);
        }))
    });
}

const index = fs.readFileSync(__dirname + '/index.html', 'utf8');

router.get('*', async function (request, response) {

    var initialState = {
        image: image,
        concepts: await getFacts(image).then(x => x)
    };
    var store = Redux.createStore(reducer, initialState);

    ReactRouter.match({
        routes: require('./routes.jsx'),
        location: request.url
    }, function (error, redirectLocation, renderProps) {
        if (renderProps) {
            var html = ReactDOMServer.renderToString(
                <Provider store={store}>
                    <ReactRouter.RouterContext {...renderProps}/>
                </Provider>
            );

            var dataHtml = index.replace(/<div id="app"><\/div>/, `<div id="app">${html}</div>`)
            response.set('Cache-Control', 'public, max-age=600, s-maxage=1200');
            response.send(dataHtml);

        } else {
            response
                .status(404)
                .send('Not Found');
        }
    });
});

module.exports = router;
