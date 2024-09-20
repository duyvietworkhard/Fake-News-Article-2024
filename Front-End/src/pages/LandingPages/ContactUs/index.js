import React, { useEffect, useState } from "react";
import Grid from "@mui/material/Grid";
import { Dialog } from "primereact/dialog";
import { connect } from "react-redux";
import PropTypes from "prop-types";
import MKBox from "components/MKBox";
import MKInput from "components/MKInput";
import MKButton from "components/MKButton";
import MKTypography from "components/MKTypography";
import "primeflex/primeflex.css";
import bgImage from "assets/images/fakeNew.png";
// import bgImage2 from "assets/images/fake-news.webp";
import checkUrl from "../../../redux/actions/checkAction"; // Sửa chính tả nếu cần
import axios from "axios";

import CircularProgress from "@mui/material/CircularProgress"; // Import CircularProgress

const ContactUs = (props) => {
  const [inputValue, setInputValue] = useState("");
  const [visible, setVisible] = useState(false);

  const closeDialog = () => setVisible(false);

  const handleChange = (event) => {
    setInputValue(event.target.value);
  };
  const [originData, setOriginData] = useState([]);
  const [translate, setTranslate] = useState(false);
  const [render, setRender] = useState(false);
  const [factData, setFactData] = useState([]);
  const [translateData, setTranslateData] = useState([]);
  const [title, setTitle] = useState([]);

  useEffect(() => {
    setTranslate(false);
    setOriginData([]);
    setFactData([]);
    setTranslateData([]);
  }, [render]);

  useEffect(() => {
    if (props.show) {
      setVisible(true);
      console.log(props.data);

      // Ensure fact_check is an array and has at least 3 elements before accessing it
      if (Array.isArray(props?.data?.fact_check) && props.data.fact_check.length >= 3) {
        setOriginData(props?.data);
        setFactData(props?.data?.fact_check?.[2]?.slice(1) || []); // Fallback to an empty array if undefined
      } else {
        setOriginData([]);
        setFactData([]);
      }
    }
  }, [props.show]);

  // Thay đổi handleCheck thành async function
  const handleCheck = async () => {
    setRender(!render);
    try {
      await props.check(inputValue); // Đợi hàm check hoàn tất
      console.log("Data from API: ", props.data);
      // setVisible(props.show); // Sau khi hoàn tất, mới setVisible
      // setOriginData(props?.data);
      // setFactData(props?.data?.fact_check?.[2]?.slice(1));
    } catch (error) {
      console.error("Error during check:", error);
    }
  };

  // const adjustUrl = (url) => {
  //   // Nếu URL chứa localhost, thay thế bằng một URL khác hoặc chỉ trả lại URL gốc
  //   if (url.startsWith("http://localhost")) {
  //     return url.replace("http://localhost", "http://yourdomain.com"); // Thay thế localhost bằng tên miền của bạn
  //   }
  //   return url;
  // };

  // Hàm dịch văn bản sử dụng Lingvanex API
  const translateText = async (text, targetLanguage = "vi") => {
    try {
      const response = await axios.post(
        "https://api-b2b.backenster.com/b1/api/v3/translate", // Endpoint của Lingvanex API
        {
          text: text,
          from: "en",
          to: targetLanguage,
        },
        {
          headers: {
            Authorization: `a_KoWf5Ysv3yG1rdaFaMEjS128X2SYT1EBzER5kvJZUHJlbdtTdCxopUAUujmeUo4IPBXGJSp3O8Wh33Pa`, // Thay thế khóa API thực tế của bạn
            "Content-Type": "application/json",
          },
        }
      );
      return response.data.result; // Đảm bảo trả về dữ liệu dịch
    } catch (error) {
      console.error("Error translating text:", error);
      return text; // Trả về văn bản gốc nếu có lỗi
    }
  };

  // Hàm xử lý dịch tất cả các đoạn văn bản
  const handleTranslate = async () => {
    const titleContent = await translateText(props?.data?.fact_check?.[2]?.[0]);
    const title = await translateText(props?.data?.fact_check?.[0]?.[1]);
    const keywordTrans = await Promise.all(
      props?.data?.fact_check?.[1]?.[1]?.map(async (keyword) => {
        try {
          const translatedText = await translateText(keyword);
          return translatedText;
        } catch (error) {
          return keyword; // Trả về văn bản gốc nếu có lỗi dịch
        }
      })
    );
    setTitle([titleContent, title, keywordTrans]);
    const newFactData = await Promise.all(
      factData.map(async (fact) => {
        const translatedFact = await Promise.all(
          fact.slice(0, 5).map(async (item) => {
            if (item && item.startsWith("http")) {
              return item; // Giữ nguyên liên kết không dịch
            }
            try {
              const translatedText = await translateText(item);
              return translatedText;
            } catch (error) {
              return item; // Trả về văn bản gốc nếu có lỗi dịch
            }
          })
        );
        return [...translatedFact, fact[5]]; // Giữ nguyên liên kết không dịch
      })
    );
    setTranslateData(newFactData);
    setTranslate(!translate);
  };

  return (
    <>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={12} lg={6}>
          <MKBox
            display={{ xs: "none", lg: "flex" }}
            width="calc(100%)"
            height="calc(100vh - 2rem)"
            borderRadius="lg"
            ml={2}
            mt={2}
            sx={{
              backgroundImage: `url(${bgImage})`,
              backgroundRepeat: "no-repeat", // Ngăn ảnh lặp lại
              backgroundSize: "contain", // Điều chỉnh kích thước ảnh
              backgroundPosition: "center",
            }}
          />
        </Grid>
        <Grid
          item
          xs={12}
          sm={10}
          md={7}
          lg={6}
          xl={4}
          ml={{ xs: "auto", lg: 6 }}
          mr={{ xs: "auto", lg: 6 }}
        >
          <MKBox
            bgColor="white"
            borderRadius="xl"
            shadow="lg"
            display="flex"
            flexDirection="column"
            justifyContent="center"
            mt={{ xs: 20, sm: 18, md: 20 }}
            mb={{ xs: 20, sm: 18, md: 20 }}
            mx={3}
          >
            <MKBox
              variant="gradient"
              bgColor="info"
              coloredShadow="info"
              borderRadius="lg"
              p={2}
              mx={2}
              mt={-3}
            >
              <MKTypography variant="h3" color="white">
                Kiểm tra thật giả
              </MKTypography>
            </MKBox>
            <MKBox p={3}>
              <MKTypography variant="body2" color="text" mb={3}>
                Trang web này hỗ trợ xác minh thông tin là thật hay giả
              </MKTypography>
              <form action="" autoComplete="off">
                <Grid container spacing={3}>
                  <Grid item xs={12} md={12}>
                    <MKInput
                      type="text"
                      variant="standard"
                      label="Nhập đường dẫn tới website bạn cần kiểm tra thông tin"
                      InputLabelProps={{ shrink: true }}
                      fullWidth
                      value={inputValue}
                      onChange={handleChange}
                    />
                  </Grid>
                </Grid>
                <Grid container item justifyContent="center" xs={12} mt={5} mb={2}>
                  <MKButton type="button" variant="gradient" color="info" onClick={handleCheck}>
                    Kiểm tra
                  </MKButton>
                </Grid>
              </form>
            </MKBox>
          </MKBox>
        </Grid>
      </Grid>
      {visible && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            backgroundColor: "rgba(0, 0, 0, 0.5)", // Lớp phủ tối
            zIndex: 1000, // Đảm bảo lớp phủ nằm dưới Dialog nhưng trên các phần tử khác
          }}
        />
      )}
      <Dialog
        visible={props.isLoading} // Sử dụng props.isLoading để hiển thị dialog
        header="Đang tải..."
        modal
        onHide={closeDialog}
        style={{
          width: "300px",
          height: "150px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "rgba(255, 255, 255, 0.9)",
          border: "none",
          borderRadius: "8px",
        }}
        className="loading-dialog"
      >
        <div style={{ textAlign: "center" }}>
          <CircularProgress /> {/* Optional: Replace this with your own loading indicator */}
          <p style={{ marginTop: "10px" }}>Chờ chút thôi...</p>
        </div>
      </Dialog>
      <Dialog
        header={
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "8px",
              backgroundImage: `url(${bgImage})`,
              backgroundSize: "contain",
              backgroundPosition: "left",
              backgroundRepeat: "no-repeat",
            }}
          >
            {/* Nội dung tiêu đề được bọc bởi một div khác */}
            <div
              style={{
                textAlign: "center", // Căn giữa nội dung tiêu đề
                flex: 1, // Chiếm toàn bộ không gian còn lại để căn giữa
                textTransform: "uppercase",
                fontWeight: "bold",
                fontSize: "60px",
                fontFamily: "Georgia, serif",
              }}
            >
              {factData.length === 0
                ? translate
                  ? "Không có kết quả"
                  : "No result"
                : !translate
                ? `${originData?.answer} NEWS`
                : originData?.answer === "Fake"
                ? "TIN GIẢ"
                : "TIN THẬT"}
            </div>

            {/* Nút button */}
            <button
              onClick={handleTranslate}
              style={{
                padding: "10px 20px",
                fontSize: "16px",
                fontWeight: "bold",
                color: "#fff",
                backgroundColor: translate ? "#007bff" : "#28a745", // Thay đổi màu nền tùy thuộc vào trạng thái
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
                transition: "background-color 0.3s ease, transform 0.2s ease",
                boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
                marginLeft: "8px", // Thêm khoảng cách 8px giữa tiêu đề và nút button
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.backgroundColor = translate ? "#0056b3" : "#218838")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.backgroundColor = translate ? "#007bff" : "#28a745")
              }
            >
              {translate ? "EN" : "VI"}
            </button>
          </div>
        }
        visible={visible}
        onHide={closeDialog}
        style={{
          width: "1200px",
          height: "800px",
          backgroundColor: "#ffffff",
          // backgroundImage: `url(${bgImage2})` /* Đường dẫn tới ảnh của bạn */,
          border: "1px solid #000",
          textAlign: "center",
          padding: "14px",
        }}
        modal
      >
        <div
          className="card"
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            height: "100%",
          }}
        >
          <div
            className="card"
            style={{
              width: "100%",
              textAlign: "center",
              backgroundColor: "#e0e0e0",
              paddingTop: "20px",
              paddingBottom: "20px",
              paddingLeft: "80px",
              paddingRight: "80px",
              border: "1px solid #000",
            }}
          >
            <h1 style={{ fontSize: "30px" }}>
              {!translate ? originData?.fact_check?.[0]?.[1] : title[1]}
            </h1>
          </div>

          {/* Chỉnh lại phần chứa từ khóa và kết quả fact-check hiển thị theo hàng dọc */}
          <div
            className="card"
            style={{
              display: "flex",
              flexDirection: "column", // Sử dụng column để xếp chồng các phần tử theo chiều dọc
              width: "100%",
              margin: "20px 0",
            }}
          >
            {/* Từ khóa */}
            <div
              className="card"
              style={{
                width: "100%", // Sử dụng 100% chiều rộng
                height: "auto", // Cho phép chiều cao tự động
                backgroundColor: "#e0e0e0",
                border: "1px solid #000",
                marginBottom: "20px", // Thêm khoảng cách dưới để tạo khoảng cách giữa các phần
                display: "flex",
                flexDirection: "column",
              }}
            >
              <label
                style={{
                  fontWeight: "bold",
                  marginBottom: "10px",
                  padding: "10px",
                  borderBottom: "1px solid #000",
                  backgroundColor: "black",
                  color: "white",
                }}
              >
                {translate ? "TỪ KHÓA" : "KEYWORDS"}
              </label>
              <div
                style={{
                  display: "flex",
                  flexDirection: "row", // Hiển thị các mục theo hàng ngang
                  justifyContent: factData.length === 0 ? "center" : "space-between", // Cách đều các mục trong hàng
                  overflowY: "auto",
                  width: "100%",
                }}
              >
                {!translate
                  ? originData?.fact_check?.[1]?.[1]?.map((item, index) => (
                      <div
                        key={index}
                        style={{
                          padding: "10px",
                          borderBottom: "1px solid #000",
                          width: "calc(100% / 3 - 20px)", // Chia đều chiều rộng cho mỗi mục, trừ đi khoảng cách padding
                          boxSizing: "border-box", // Đảm bảo padding không làm ảnh hưởng tới chiều rộng
                          fontFamily: "Times New Roman, serif", // Font chữ báo chí
                          fontSize: "16px", // Cỡ chữ mục từ khóa
                        }}
                      >
                        {item}
                      </div>
                    ))
                  : title[2].map((item, index) => (
                      <div
                        key={index}
                        style={{
                          padding: "10px",
                          borderBottom: "1px solid #000",
                          width: "calc(100% / 3 - 20px)", // Chia đều chiều rộng cho mỗi mục, trừ đi khoảng cách padding
                          boxSizing: "border-box", // Đảm bảo padding không làm ảnh hưởng tới chiều rộng
                          fontFamily: "Times New Roman, serif", // Font chữ báo chí
                          fontSize: "16px", // Cỡ chữ mục từ khóa
                        }}
                      >
                        {item}
                      </div>
                    ))}
              </div>
            </div>

            {/* Kết quả fact-check */}
            <div
              className="card"
              style={{
                width: "100%", // Sử dụng 100% chiều rộng
                padding: "20px",
                backgroundColor: "#f9f9f9",
                border: "1px solid #ccc",
                borderRadius: "8px",
                boxShadow: "0px 4px 8px rgba(0, 0, 0, 0.1)",
                overflowY: "auto",
                maxHeight: "50vh", // Đặt chiều cao tối đa để hỗ trợ cuộn
              }}
            >
              <h2 style={{ textAlign: "center", marginBottom: "20px" }}>
                {!translate ? originData?.fact_check?.[2]?.[0] : title[0]}
              </h2>
              {translate && Array.isArray(translateData) && translateData.length > 0 ? (
                translateData.map((fact, index) => (
                  <div
                    key={index}
                    style={{
                      marginBottom: "20px",
                      padding: "15px",
                      backgroundColor: "#fff",
                      borderRadius: "8px",
                      border: "1px solid #ddd",
                      fontFamily: "Times New Roman, serif",
                      fontSize: "16px",
                    }}
                  >
                    <p style={{ fontWeight: "bold", marginBottom: "10px" }}>{fact?.[0]}</p>
                    <p style={{ marginBottom: "10px" }}>
                      <strong>{fact[1]}</strong> {fact?.[2]}
                    </p>
                    <p style={{ marginBottom: "10px" }}>
                      <strong>{fact[3]}</strong> {fact?.[4]}
                    </p>
                    <a
                      href={
                        fact[5]
                          .replace(/^\(|\)$/, "")
                          .replace(/[\s()]+$/, "")
                          .startsWith("http")
                          ? fact[5].replace(/^\(|\)$/, "").replace(/[\s()]+$/, "")
                          : `https://${fact[5].replace(/^\(|\)$/, "").replace(/[\s()]+$/, "")}`
                      }
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: "#007bff", textDecoration: "underline" }}
                    >
                      Đi tới liên kết
                    </a>
                  </div>
                ))
              ) : Array.isArray(factData) && factData.length > 0 ? (
                factData.map((fact, index) => (
                  <div
                    key={index}
                    style={{
                      marginBottom: "20px",
                      padding: "15px",
                      backgroundColor: "#fff",
                      borderRadius: "8px",
                      border: "1px solid #ddd",
                      fontFamily: "Times New Roman, serif",
                      fontSize: "16px",
                    }}
                  >
                    <p style={{ fontWeight: "bold", marginBottom: "10px" }}>{fact?.[0]}</p>
                    <p style={{ marginBottom: "10px" }}>
                      <strong>{fact[1]}</strong> {fact?.[2]}
                    </p>
                    <p style={{ marginBottom: "10px" }}>
                      <strong>{fact[3]}</strong> {fact?.[4]}
                    </p>
                    <a
                      href={
                        fact[5]
                          .replace(/^\(|\)$/, "")
                          .replace(/[\s()]+$/, "")
                          .startsWith("http")
                          ? fact[5].replace(/^\(|\)$/, "").replace(/[\s()]+$/, "")
                          : `https://${fact[5].replace(/^\(|\)$/, "").replace(/[\s()]+$/, "")}`
                      }
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: "#007bff", textDecoration: "underline" }}
                    >
                      {translate ? "Đi tới liên kết" : "Go to the link"}
                    </a>
                  </div>
                ))
              ) : (
                // <div>{!translate ? "No result" : "Không có kết quả"}</div>
                <div></div>
              )}
            </div>
          </div>

          <div
            className="card"
            style={{
              width: "100%",
              height: "150px",
              backgroundColor: "#e0e0e0",
              border: "1px solid #000",
            }}
          ></div>
        </div>
      </Dialog>
    </>
  );
};

ContactUs.propTypes = {
  check: PropTypes.func.isRequired,
  isLoading: PropTypes.bool.isRequired,
  show: PropTypes.bool.isRequired,
  data: PropTypes.object,
};

const mapStateToProps = (state) => ({
  isLoading: state.check.isLoading,
  data: state.check.data,
  show: state?.check?.show,
});

const mapDispatchToProps = (dispatch) => ({
  check: (url) => dispatch(checkUrl(url)), // Sửa từ chechUrl thành checkUrl
});

export default connect(mapStateToProps, mapDispatchToProps)(ContactUs);
