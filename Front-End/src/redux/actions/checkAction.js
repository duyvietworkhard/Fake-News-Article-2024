import axios from "axios";
const CHECK_REQUEST = "CHECK_REQUEST";
const CHECK_SUCCESS = "CHECK_SUCCESS";
const CHECK_FAIL = "CHECK_FAIL";

const checkUrl = (url) => async (dispatch) => {
  dispatch({
    type: CHECK_REQUEST,
    payload: null,
  });

  try {
    const response = await axios.post("answer", {
      url,
    });
    dispatch({
      type: CHECK_SUCCESS,
      payload: response.data,
    });
  } catch (error) {
    dispatch({
      type: CHECK_FAIL,
      payload: error.response.data,
    });

    alert(error.response.data.message);
  }
};

export default checkUrl;
