import React, {Component} from 'react';
import Helmet from 'react-helmet';

import {config} from 'config';
import PageLink from '../components/common/PageLink';
import Clarifai from 'clarifai';
import {isNullOrUndefined} from 'util';
import _ from 'underscore';

const app = new Clarifai.App({apiKey: '4f69d7da58aa458baa84be2c3a56aff5'});

let image = 'https://picsum.photos/500/500?image=' + Math.floor(Math.random() * Math.floor(1000)).toString();

class IndexPage extends Component {

    constructor(props) {
        super(props);
        this.state = {
            descriptors: []
        };
    }

    componentDidMount() {
        this.descriptors()       
    }

    descriptors() {
        app
            .models
            .initModel({id: Clarifai.GENERAL_MODEL, version: "aa7f35c01e0642fda5cf400f543e7c40"})
            .then(generalModel => {
                return generalModel.predict(image);
            })
            .then(response => {
                let desc = [...response['outputs'][0]['data']['concepts'].map(x => x.name)]
                this.setState({
                    descriptors: desc,
                })
            })
            .catch((err) => {
                console.log(err);
            });
    }

    render() {
        return (
            <div style={{
                width: '100%'
            }}>
                <div
                    style={{
                    display: 'inline',
                    width: '50px'
                }}><img src={image + '&blur'}/></div>
                <div
                    style={{
                    display: 'inline',
                    width: '50px'
                }}>
                    <p>What kind of {this.state.descriptors[0]} is in the photo?</p>
                </div>
            </div>
        )
    }
}

export default IndexPage;
