//! PyO3 bindings for the Polymarket Rust CLOB client SDK.
//!
//! Exposes `RustClobClient` as a Python class with async methods that
//! bridge tokio futures to Python awaitables via `pyo3-async-runtimes`.

use std::str::FromStr;
use std::sync::Arc;

use alloy::signers::local::PrivateKeySigner;
use alloy::signers::Signer as _;
use chrono::DateTime;
use polymarket_client_sdk::clob::types::request::{
    BalanceAllowanceRequest, OrdersRequest,
};
use polymarket_client_sdk::clob::types::{OrderType, Side, SignatureType};
use polymarket_client_sdk::clob::{Client, Config};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rust_decimal::Decimal;
use tokio::runtime::Runtime;

/// Type alias for the authenticated client with Normal auth kind.
type AuthClient = Client<
    polymarket_client_sdk::auth::state::Authenticated<polymarket_client_sdk::auth::Normal>,
>;

/// Internal state shared across async calls.
struct ClientInner {
    client: AuthClient,
    signer: PrivateKeySigner,
}

// Safety: Client and PrivateKeySigner are Send + Sync already.
unsafe impl Send for ClientInner {}
unsafe impl Sync for ClientInner {}

/// A Polymarket CLOB client backed by the native Rust SDK.
#[pyclass]
struct RustClobClient {
    inner: Arc<ClientInner>,
}

fn to_py_err(e: polymarket_client_sdk::error::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

fn parse_order_type(s: &str) -> Result<OrderType, PyErr> {
    match s.to_uppercase().as_str() {
        "GTC" => Ok(OrderType::GTC),
        "FOK" => Ok(OrderType::FOK),
        "FAK" => Ok(OrderType::FAK),
        "GTD" => Ok(OrderType::GTD),
        other => Err(PyRuntimeError::new_err(format!("Unknown order type: {other}"))),
    }
}

fn parse_side(s: &str) -> Result<Side, PyErr> {
    match s.to_uppercase().as_str() {
        "BUY" => Ok(Side::Buy),
        "SELL" => Ok(Side::Sell),
        other => Err(PyRuntimeError::new_err(format!("Unknown side: {other}"))),
    }
}

#[pymethods]
impl RustClobClient {
    /// Create a new authenticated Rust CLOB client.
    #[new]
    #[pyo3(signature = (host, key, chain_id, funder=None, signature_type=2))]
    fn new(
        host: String,
        key: String,
        chain_id: u64,
        funder: Option<String>,
        signature_type: u8,
    ) -> PyResult<Self> {
        let rt = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;

        let signer = PrivateKeySigner::from_str(&key)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid private key: {e}")))?
            .with_chain_id(Some(chain_id));

        let sig_type = match signature_type {
            0 => SignatureType::Eoa,
            1 => SignatureType::Proxy,
            2 => SignatureType::GnosisSafe,
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "invalid signature_type: {other}"
                )))
            }
        };

        let client = rt.block_on(async {
            let config = Config::builder().use_server_time(true).build();
            let unauthenticated = Client::new(&host, config).map_err(to_py_err)?;

            let mut auth_builder = unauthenticated.authentication_builder(&signer);
            auth_builder = auth_builder.signature_type(sig_type);

            if let Some(ref funder_addr) = funder {
                let addr = alloy::primitives::Address::from_str(funder_addr)
                    .map_err(|e| PyRuntimeError::new_err(format!("invalid funder: {e}")))?;
                auth_builder = auth_builder.funder(addr);
            }

            auth_builder.authenticate().await.map_err(to_py_err)
        })?;

        Ok(Self {
            inner: Arc::new(ClientInner { client, signer }),
        })
    }

    /// Get the backend name (always "rust").
    fn backend(&self) -> &str {
        "rust"
    }

    /// Create, sign, and post a limit order in one Rust call.
    #[pyo3(signature = (token_id, price, size, side, order_type, expiration=None))]
    fn create_and_post_order<'py>(
        &self,
        py: Python<'py>,
        token_id: String,
        price: String,
        size: String,
        side: String,
        order_type: String,
        expiration: Option<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py_with_locals::<_, PyObject>(
            py,
            pyo3_async_runtimes::tokio::get_current_locals(py)?,
            async move {
                let token_id_u256 = alloy::primitives::U256::from_str(&token_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("invalid token_id: {e}")))?;
                let price_dec = Decimal::from_str(&price)
                    .map_err(|e| PyRuntimeError::new_err(format!("invalid price: {e}")))?;
                let size_dec = Decimal::from_str(&size)
                    .map_err(|e| PyRuntimeError::new_err(format!("invalid size: {e}")))?;
                let side_enum = parse_side(&side)?;
                let ot = parse_order_type(&order_type)?;

                let mut builder = inner.client.limit_order()
                    .token_id(token_id_u256)
                    .price(price_dec)
                    .size(size_dec)
                    .side(side_enum)
                    .order_type(ot);

                if let Some(exp) = expiration {
                    if exp > 0 {
                        let dt = DateTime::from_timestamp(exp, 0)
                            .ok_or_else(|| PyRuntimeError::new_err("invalid expiration"))?;
                        builder = builder.expiration(dt);
                    }
                }

                let signable = builder.build().await.map_err(to_py_err)?;
                let signed = inner.client.sign(&inner.signer, signable).await.map_err(to_py_err)?;
                let response = inner.client.post_order(signed).await.map_err(to_py_err)?;

                Python::with_gil(|py| {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("orderID", &response.order_id)?;
                    dict.set_item("success", response.success)?;
                    dict.set_item("status", format!("{:?}", response.status))?;
                    if let Some(ref err) = response.error_msg {
                        dict.set_item("errorMsg", err)?;
                    }
                    Ok(dict.unbind().into())
                })
            },
        )
    }

    /// Fetch order book for a token.
    fn get_order_book<'py>(
        &self,
        py: Python<'py>,
        token_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py_with_locals::<_, PyObject>(
            py,
            pyo3_async_runtimes::tokio::get_current_locals(py)?,
            async move {
                let token_id_u256 = alloy::primitives::U256::from_str(&token_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("invalid token_id: {e}")))?;

                let request = polymarket_client_sdk::clob::types::request::OrderBookSummaryRequest::builder()
                    .token_id(token_id_u256)
                    .build();

                let book = inner.client.order_book(&request).await.map_err(to_py_err)?;

                Python::with_gil(|py| {
                    let dict = pyo3::types::PyDict::new(py);
                    let bids = pyo3::types::PyList::empty(py);
                    for bid in &book.bids {
                        let d = pyo3::types::PyDict::new(py);
                        d.set_item("price", bid.price.to_string())?;
                        d.set_item("size", bid.size.to_string())?;
                        bids.append(d)?;
                    }
                    let asks = pyo3::types::PyList::empty(py);
                    for ask in &book.asks {
                        let d = pyo3::types::PyDict::new(py);
                        d.set_item("price", ask.price.to_string())?;
                        d.set_item("size", ask.size.to_string())?;
                        asks.append(d)?;
                    }
                    dict.set_item("bids", bids)?;
                    dict.set_item("asks", asks)?;
                    dict.set_item("asset_id", book.asset_id.to_string())?;
                    Ok(dict.unbind().into())
                })
            },
        )
    }

    /// Cancel an order by exchange order ID.
    fn cancel<'py>(
        &self,
        py: Python<'py>,
        order_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py_with_locals::<_, PyObject>(
            py,
            pyo3_async_runtimes::tokio::get_current_locals(py)?,
            async move {
                let resp = inner.client.cancel_order(&order_id).await.map_err(to_py_err)?;

                Python::with_gil(|py| {
                    let dict = pyo3::types::PyDict::new(py);
                    let canceled = pyo3::types::PyList::empty(py);
                    for id in &resp.canceled {
                        canceled.append(id)?;
                    }
                    dict.set_item("canceled", canceled)?;
                    Ok(dict.unbind().into())
                })
            },
        )
    }

    /// Get a single order by exchange ID.
    fn get_order<'py>(
        &self,
        py: Python<'py>,
        order_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py_with_locals::<_, PyObject>(
            py,
            pyo3_async_runtimes::tokio::get_current_locals(py)?,
            async move {
                let order = inner.client.order(&order_id).await.map_err(to_py_err)?;

                Python::with_gil(|py| {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("id", &order.id)?;
                    dict.set_item("status", format!("{:?}", order.status))?;
                    dict.set_item("side", format!("{:?}", order.side))?;
                    dict.set_item("price", order.price.to_string())?;
                    dict.set_item("original_size", order.original_size.to_string())?;
                    dict.set_item("size_matched", order.size_matched.to_string())?;
                    dict.set_item("outcome", &order.outcome)?;
                    dict.set_item("order_type", format!("{:?}", order.order_type))?;
                    Ok(dict.unbind().into())
                })
            },
        )
    }

    /// Get all open orders for the authenticated user.
    fn get_orders<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py_with_locals::<_, PyObject>(
            py,
            pyo3_async_runtimes::tokio::get_current_locals(py)?,
            async move {
                let page = inner.client
                    .orders(&OrdersRequest::default(), None)
                    .await
                    .map_err(to_py_err)?;

                Python::with_gil(|py| {
                    let list = pyo3::types::PyList::empty(py);
                    for order in &page.data {
                        let d = pyo3::types::PyDict::new(py);
                        d.set_item("id", &order.id)?;
                        d.set_item("status", format!("{:?}", order.status))?;
                        d.set_item("side", format!("{:?}", order.side))?;
                        d.set_item("price", order.price.to_string())?;
                        d.set_item("original_size", order.original_size.to_string())?;
                        d.set_item("size_matched", order.size_matched.to_string())?;
                        list.append(d)?;
                    }
                    Ok(list.unbind().into())
                })
            },
        )
    }

    /// Get USDC balance and allowances.
    fn get_balance_allowance<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py_with_locals::<_, PyObject>(
            py,
            pyo3_async_runtimes::tokio::get_current_locals(py)?,
            async move {
                let resp = inner.client
                    .balance_allowance(BalanceAllowanceRequest::default())
                    .await
                    .map_err(to_py_err)?;

                Python::with_gil(|py| {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("balance", resp.balance.to_string())?;
                    Ok(dict.unbind().into())
                })
            },
        )
    }

    /// Get midpoint price for a token.
    fn get_midpoint<'py>(
        &self,
        py: Python<'py>,
        token_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py_with_locals::<_, PyObject>(
            py,
            pyo3_async_runtimes::tokio::get_current_locals(py)?,
            async move {
                let token_id_u256 = alloy::primitives::U256::from_str(&token_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("invalid token_id: {e}")))?;

                let request = polymarket_client_sdk::clob::types::request::MidpointRequest::builder()
                    .token_id(token_id_u256)
                    .build();

                let resp = inner.client.midpoint(&request).await.map_err(to_py_err)?;

                Python::with_gil(|py| {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("mid", resp.mid.to_string())?;
                    Ok(dict.unbind().into())
                })
            },
        )
    }
}

#[pymodule]
fn rs_clob_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustClobClient>()?;
    Ok(())
}
