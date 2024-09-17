import { applyMiddleware, combineReducers, compose, createStore } from "redux";
import { thunk } from "redux-thunk";
import checkReducer from "./redux/reducers/checkReducer";

const rootReduct = combineReducers({
  check: checkReducer,
});

const composeEnhancer = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;
const store = createStore(rootReduct, composeEnhancer(applyMiddleware(thunk)));

export default store;
