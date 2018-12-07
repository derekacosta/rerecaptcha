import React from 'react';
import '../css/main';

import PageLink from '../components/common/PageLink';

module.exports = React.createClass({
    propTypes() {
        return {children: React.PropTypes.any};
    },
    render() {
        return (
            <div>
                <div className='site-header'>
                    <p>re-reCAPTCHA</p>
                </div>
                <div>
                    {this.props.children}
                </div>
            </div>
        );
    }
});
