import React from 'react';
import {Link} from 'react-router';
import {connect} from 'react-redux';

class Layout extends React.Component {
    constructor() {
        super();
        this._handleClick = this
            ._handleClick
            .bind(this);
    }
    _handleClick() {
        console.log(this.props.custom.concepts);
    }
    render() {
        const {custom} = this.props;
        console.log(custom);

        return (
            <div>
                <h1>rerereCAPTCHA</h1>
                <img src={custom.image}/>
                <button onClick={this._handleClick}>Click Me</button>
                {this.props.children}
                <ul>
                    <li>
                        <Link to='/'>Home</Link>
                    </li>
                    <li>
                        <Link to='/about'>About</Link>
                    </li>
                </ul>
                <script
                    dangerouslySetInnerHTML={{
                    __html: 'window.PROPS=' + JSON.stringify(custom)
                }}/>
            </div>
        );
    }
}

const wrapper = connect((state) => {
    return {custom: state};
});

export default wrapper(Layout);
