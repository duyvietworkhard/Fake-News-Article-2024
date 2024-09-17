const CHECK_REQUEST = "CHECK_REQUEST";
const CHECK_SUCCESS = "CHECK_SUCCESS";
const CHECK_FAIL = "CHECK_FAIL";
const initialState = {
  isLoading: false,
  data: [],
  show: false,
};

const checkReducer = (state = initialState, action) => {
  switch (action.type) {
    //signin
    case CHECK_REQUEST:
      return {
        ...state,
        isLoading: true,
        show: false,
      };
    case CHECK_SUCCESS:
      return {
        ...state,
        isLoading: false,
        data: action.payload,
        show: true,
      };
    case CHECK_FAIL:
      return {
        ...state,
        isLoading: false,
        show: false,
        signinErr: action.payload,
      };
    default:
      return state;
  }
};

export default checkReducer;
