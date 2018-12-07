import React, {Component} from 'react';
import Helmet from 'react-helmet';

import {config} from 'config';
import PageLink from '../components/common/PageLink';
import Clarifai from 'clarifai';
import {isNullOrUndefined} from 'util';

const app = new Clarifai.App({apiKey: '22cc6a81184f43dc98350f1b97907e6c'});

let image = 'https://picsum.photos/500/500?image=' + Math.floor(Math.random() * Math.floor(1000)).toString();

let concepts;

app
    .models
    .initModel({id: Clarifai.GENERAL_MODEL, version: "aa7f35c01e0642fda5cf400f543e7c40"})
    .then(generalModel => {
        return generalModel.predict(image);
    })
    .then(response => {
        console.log(response['outputs'][0]['data']['concepts']);
    })

// concepts = concepts[0].name;
    
class IndexPage extends Component {
    render() {
        return (
            <div style={{
                width: '100%'
            }}>
                <div
                    style={{
                    display: 'inline',
                    width: '50px'
                }}><img src={image}/></div>
                <div
                    style={{
                    display: 'inline',
                    width: '50px'
                }}>
                  <p>What kind of {concepts} is in the photo?</p>
                </div>
            </div>
        )
    }
}

export default IndexPage;
