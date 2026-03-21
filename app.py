#!/usr/bin/env python
"""
ES Futures Quantitative Research Platform
FIXED: Shows COMPLETE CME sessions from 18:00 to 17:00 next day without cutting
Fixed timestamp error in vline annotations - REMOVED ALL VERTICAL LINES
"""

import sys
import argparse
import webbrowser
from threading import Timer
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union, Literal
import time

# Third-party imports
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from scipy import stats
from scipy.stats import gaussian_kde
from loguru import logger
import pytz
import joblib

# HMM imports
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. HMM features disabled.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dark theme for plotly
load_figure_template("darkly")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Main configuration class for the research platform."""
    
    # Data configuration
    ticker: str = "ES=F"
    data_resolution: Literal["1m", "5m"] = "5m"
    data_cache_dir: Path = Path("./data/cache")
    
    # Session configuration - FIXED: Proper CME session (18:00 to 17:00 next day)
    session_start_utc: int = 18  # 18:00 UTC-5
    session_end_utc: int = 17    # 17:00 UTC-5 next day
    timezone: str = "America/New_York"
    
    # Statistical thresholds
    z_score_thresholds: list[float] = None
    percentile_thresholds: list[float] = None
    
    # Volatility regime configuration
    rolling_window: int = 20
    volatility_regime_thresholds: dict[str, float] = None
    
    # HMM configuration
    hmm_n_states: int = 4
    hmm_n_iter: int = 100
    
    # Dashboard configuration
    dashboard_host: str = "localhost"
    dashboard_port: int = 8050
    debug_mode: bool = False
    
    # Performance
    use_caching: bool = True
    max_cache_size_gb: float = 2.0
    
    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.z_score_thresholds is None:
            self.z_score_thresholds = [2.0, 3.0, 4.0, 5.0]
        
        if self.percentile_thresholds is None:
            self.percentile_thresholds = [1.0, 5.0, 95.0, 99.0]
        
        if self.volatility_regime_thresholds is None:
            self.volatility_regime_thresholds = {
                "low": 0.25,
                "normal": 0.5,
                "high": 0.75,
                "extreme": 0.95
            }
        
        # Create directories
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        Path("./data/levels").mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)
        Path("./data/hmm").mkdir(parents=True, exist_ok=True)


# Global configuration
config = Config()


# ============================================================================
# UTILITIES
# ============================================================================

def setup_logging():
    """Configure logging for the application."""
    logger.add(
        "logs/es_quant_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )


def convert_to_utc5(dt: pd.Timestamp) -> pd.Timestamp:
    """Convert timestamp to UTC-5 (CME reference time)."""
    if dt.tz is None:
        dt = dt.tz_localize('UTC')
    return dt.tz_convert('America/New_York')


def get_full_session_range(session_date: datetime) -> Tuple[datetime, datetime]:
    """
    Get the full CME session range for a given session date.
    Session runs from 18:00 on session_date to 17:00 next day.
    
    Args:
        session_date: The date of the session (e.g., 2026-03-08)
        
    Returns:
        Tuple of (session_start, session_end)
    """
    # Session starts at 18:00 on the session date
    session_start = session_date.replace(hour=18, minute=0, second=0, microsecond=0)
    
    # Session ends at 17:00 next day
    session_end = session_start + timedelta(hours=23)  # 23 hours later = 17:00 next day
    
    return session_start, session_end


def calculate_session_id(dt: pd.Timestamp) -> str:
    """Calculate CME session ID for a given timestamp."""
    dt_utc5 = convert_to_utc5(dt)
    
    # A session runs from 18:00 to 17:00 next day
    # So for any given time, the session ID is the date of the session start (18:00)
    if dt_utc5.hour >= 18:
        # After 18:00, session started today at 18:00
        session_date = dt_utc5.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # Before 18:00, session started yesterday at 18:00
        session_date = (dt_utc5 - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    return session_date.strftime('%Y-%m-%d')


def format_session_range(session_start: datetime, session_end: datetime) -> str:
    """Format session range for display."""
    start_str = session_start.strftime('%Y-%m-%d %H:%M')
    end_str = session_end.strftime('%Y-%m-%d %H:%M')
    return f"{start_str} to {end_str}"


def cache_key(func_name: str, *args, **kwargs) -> str:
    """Generate cache key for function calls."""
    key_parts = [func_name]
    key_parts.extend([str(arg) for arg in args])
    key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
    key_string = "_".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached_call(cache_dir: Path, key: str, func, *args, **kwargs):
    """Cached function call with joblib."""
    if not config.use_caching:
        return func(*args, **kwargs)
    
    cache_file = cache_dir / f"{key}.joblib"
    
    if cache_file.exists():
        logger.debug(f"Loading from cache: {cache_file}")
        return joblib.load(cache_file)
    
    logger.debug(f"Computing and caching: {cache_file}")
    result = func(*args, **kwargs)
    joblib.dump(result, cache_file)
    return result


# ============================================================================
# HIDDEN MARKOV MODEL REGIME DETECTOR
# ============================================================================

class HMMRegimeDetector:
    """Hidden Markov Model for sophisticated regime detection."""
    
    def __init__(self):
        self.hmm_dir = Path("./data/hmm")
        self.hmm_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        
    def fit_hmm(self, returns: np.ndarray, session_id: str, n_states: int = None) -> Dict[str, Any]:
        """Fit Hidden Markov Model to returns."""
        if not HMM_AVAILABLE:
            return {'error': 'hmmlearn not installed'}
        
        if n_states is None:
            n_states = config.hmm_n_states
        
        if len(returns) < 50:
            return {'error': 'Insufficient data'}
        
        try:
            # Reshape for HMM
            X = returns.reshape(-1, 1)
            
            # Fit model with multiple attempts
            best_model = None
            best_score = -np.inf
            
            for _ in range(3):  # Try 3 times with different initializations
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=config.hmm_n_iter,
                    random_state=42,
                    init_params="stmc",
                    params="stmc"
                )
                
                try:
                    model.fit(X)
                    score = model.score(X)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except:
                    continue
            
            if best_model is None:
                return {'error': 'Failed to fit HMM'}
            
            # Predict states
            states = best_model.predict(X)
            
            # Get state parameters
            state_means = best_model.means_.flatten()
            state_vars = np.sqrt(best_model.covars_).flatten()
            
            # Order states by variance (volatility)
            state_order = np.argsort(state_vars)
            
            # Calculate state statistics
            state_stats = []
            for i, state_idx in enumerate(state_order):
                mask = states == state_idx
                state_stats.append({
                    'state': i,
                    'original_idx': int(state_idx),
                    'mean': float(state_means[state_idx]),
                    'vol': float(state_vars[state_idx]),
                    'proportion': float(np.mean(mask)),
                    'duration': self._calculate_mean_duration(states == state_idx)
                })
            
            # Calculate transition matrix
            transmat = best_model.transmat_[state_order][:, state_order]
            
            # Calculate regime persistence
            persistence = np.trace(transmat) / n_states
            
            # Save model
            model_path = self.hmm_dir / f"hmm_{session_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            # Save to cache
            self.models[session_id] = best_model
            
            return {
                'states': states.tolist(),
                'state_stats': state_stats,
                'transition_matrix': transmat.tolist(),
                'persistence': float(persistence),
                'n_states': n_states,
                'log_likelihood': float(best_score),
                'state_order': state_order.tolist()
            }
            
        except Exception as e:
            logger.error(f"HMM fitting error: {e}")
            return {'error': str(e)}
    
    def _calculate_mean_duration(self, state_mask: np.ndarray) -> float:
        """Calculate mean duration in a state."""
        # Find state transitions
        transitions = np.diff(state_mask.astype(int))
        start_idx = np.where(transitions == 1)[0] + 1
        end_idx = np.where(transitions == -1)[0] + 1
        
        if len(start_idx) == 0 or len(end_idx) == 0:
            return float(len(state_mask))
        
        # Align starts and ends
        if start_idx[0] > end_idx[0]:
            end_idx = end_idx[1:]
        if len(start_idx) > len(end_idx):
            start_idx = start_idx[:len(end_idx)]
        
        durations = end_idx - start_idx
        return float(np.mean(durations)) if len(durations) > 0 else 0
    
    def get_current_regime(self, session_id: str, returns: np.ndarray) -> Dict[str, Any]:
        """Get current HMM regime."""
        if session_id not in self.models:
            return {}
        
        model = self.models[session_id]
        X = returns.reshape(-1, 1)
        states = model.predict(X)
        
        # Get latest state
        current_state = int(states[-1])
        
        # Get state parameters
        state_means = model.means_.flatten()
        state_vars = np.sqrt(model.covars_).flatten()
        
        # Order by volatility
        state_order = np.argsort(state_vars)
        state_names = ['Low Vol', 'Normal Vol', 'High Vol', 'Extreme Vol'][:len(state_order)]
        
        # Map current state to named regime
        current_regime_idx = np.where(state_order == current_state)[0][0]
        current_regime = state_names[current_regime_idx]
        
        return {
            'current_state': current_state,
            'current_regime': current_regime,
            'state_vol': float(state_vars[current_state]),
            'state_mean': float(state_means[current_state]),
            'regime_probs': model.predict_proba(X)[-1].tolist()
        }
    
    def visualize_transitions(self, session_id: str) -> go.Figure:
        """Create transition matrix visualization."""
        if session_id not in self.models:
            return go.Figure()
        
        model = self.models[session_id]
        
        # Order states by volatility
        state_vars = np.sqrt(model.covars_).flatten()
        state_order = np.argsort(state_vars)
        
        # Create ordered transition matrix
        transmat = model.transmat_[state_order][:, state_order]
        
        # State names
        state_names = ['Low Vol', 'Normal Vol', 'High Vol', 'Extreme Vol'][:len(state_order)]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=transmat,
            x=state_names,
            y=state_names,
            colorscale='Viridis',
            text=np.round(transmat, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(title="Probability")
        ))
        
        fig.update_layout(
            template="plotly_dark",
            title="HMM State Transition Matrix",
            xaxis_title="To State",
            yaxis_title="From State",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig


# ============================================================================
# DATA FETCHER - FIXED FOR COMPLETE SESSIONS
# ============================================================================

class DataFetcher:
    """Handles downloading and caching of ES futures data."""
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.ticker = config.ticker
        self.resolution = config.data_resolution
        self.cache_dir = config.data_cache_dir
        
    def fetch_data(self, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   force_refresh: bool = False) -> pd.DataFrame:
        """Fetch ES futures data from yfinance, handling the 60-day limit."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # Default to 30 days
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # For 5-minute data, yfinance has a 60-day limit
        if self.resolution == "5m":
            return self._fetch_chunked_data(start_date, end_date)
        else:
            # For 1-minute data, use chunked approach as well
            return self._fetch_chunked_data(start_date, end_date)
    
    def _fetch_chunked_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data in chunks to handle yfinance limits."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Add buffer to ensure we get complete sessions
        start = start - timedelta(days=2)
        end = end + timedelta(days=2)
        
        max_chunk_days = 30  # Conservative: 30 days per chunk
        
        all_dfs = []
        current_start = start
        
        logger.info(f"Fetching data in chunks of {max_chunk_days} days...")
        
        while current_start < end:
            current_end = min(current_start + timedelta(days=max_chunk_days), end)
            
            chunk_start = current_start.strftime('%Y-%m-%d')
            chunk_end = current_end.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching chunk: {chunk_start} to {chunk_end}")
            
            cache_key_str = cache_key(
                "fetch_chunk", 
                self.ticker, 
                self.resolution, 
                chunk_start, 
                chunk_end
            )
            
            chunk_df = cached_call(
                self.cache_dir,
                cache_key_str,
                self._download_data,
                chunk_start,
                chunk_end
            )
            
            if not chunk_df.empty:
                all_dfs.append(chunk_df)
            
            current_start = current_end
            time.sleep(0.5)  # Be nice to Yahoo's servers
        
        if not all_dfs:
            raise ValueError("No data returned from yfinance for any chunk")
        
        # Combine all chunks
        final_df = pd.concat(all_dfs)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df = final_df.sort_index()
        
        logger.info(f"Total data fetched: {len(final_df)} rows from {final_df.index[0]} to {final_df.index[-1]}")
        
        return final_df
    
    def _download_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download a single chunk of data from yfinance."""
        logger.info(f"Downloading {self.ticker} chunk: {start_date} to {end_date}")
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Pauza před každým requestem - zabrání rate limitu
                time.sleep(3)
                
                ticker = yf.Ticker(self.ticker)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=self.resolution,
                    actions=False
                )
                
                if df.empty:
                    return pd.DataFrame()
                
                df = self._prepare_dataframe(df)
                return df
                
            except Exception as e:
                if "Too Many Requests" in str(e) or "Rate" in str(e) or "429" in str(e):
                    wait = (attempt + 1) * 20  
                    time.sleep(wait)
                else:
                    return pd.DataFrame()
        
        return pd.DataFrame()
        

    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean the downloaded dataframe."""
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Convert to NY time
        df.index = df.index.tz_convert('America/New_York')
        
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in columns_to_keep if col in df.columns]]
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = df['high'] - df['low']
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        df = df.dropna()
        return df


# ============================================================================
# SESSION MANAGER - FIXED FOR COMPLETE SESSIONS
# ============================================================================

class SessionManager:
    """Manages CME trading sessions for ES futures."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.session_start = config.session_start_utc
        self.session_end = config.session_end_utc
        self.timezone = config.timezone
        self.sessions: Dict[str, pd.DataFrame] = {}
        
    def create_sessions(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into individual CME trading sessions (18:00 to 17:00 next day)."""
        if df is None or df.empty:
            logger.error("Invalid dataframe for session creation")
            return {}
        
        logger.info("Creating CME trading sessions (18:00 → 17:00 next day)")
        
        # First, add session IDs to all rows
        df_with_sessions = df.copy()
        df_with_sessions['session_id'] = df_with_sessions.index.map(calculate_session_id)
        
        # Group by session_id
        sessions = {}
        for session_id, session_data in df_with_sessions.groupby('session_id'):
            session_data = session_data.sort_index()
            
            # Parse session date
            session_date = datetime.strptime(session_id, '%Y-%m-%d')
            session_date = pytz.timezone(self.timezone).localize(session_date)
            
            # Get the full session range for this session
            session_start, session_end = get_full_session_range(session_date)
            
            # Filter data to exact session range
            session_data = session_data[(session_data.index >= session_start) & 
                                       (session_data.index <= session_end)]
            
            # Only keep sessions with enough data
            if not session_data.empty and len(session_data) > 10:
                sessions[session_id] = session_data
                logger.info(f"✅ Created COMPLETE session {session_id}: "
                          f"{session_start.strftime('%Y-%m-%d %H:%M')} to "
                          f"{session_end.strftime('%Y-%m-%d %H:%M')} "
                          f"({len(session_data)} rows)")
        
        self.sessions = sessions
        logger.info(f"Created {len(sessions)} complete trading sessions")
        
        # Log session ranges for verification
        for session_id, session_data in sessions.items():
            start = session_data.index[0]
            end = session_data.index[-1]
            expected_start, expected_end = get_full_session_range(
                datetime.strptime(session_id, '%Y-%m-%d').replace(tzinfo=pytz.timezone(self.timezone))
            )
            
            logger.info(f"  Session {session_id}:")
            logger.info(f"    Expected: {expected_start.strftime('%Y-%m-%d %H:%M')} to {expected_end.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"    Actual:   {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
            
            # Check if session is complete (allow 1 hour margin for data availability)
            if abs((start - expected_start).total_seconds()) < 3600 and abs((end - expected_end).total_seconds()) < 3600:
                logger.info(f"    ✅ Session COMPLETE")
            else:
                logger.warning(f"    ⚠️ Session INCOMPLETE - missing data")
        
        return sessions
    
    def get_current_session(self) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """Get the current trading session."""
        now = datetime.now(pytz.timezone(self.timezone))
        session_id = calculate_session_id(now)
        
        if session_id in self.sessions:
            return session_id, self.sessions[session_id]
        return None, None


# ============================================================================
# STATISTICS ENGINE
# ============================================================================

class StatisticsEngine:
    """Computes statistical metrics and levels for trading sessions."""
    
    def __init__(self):
        """Initialize the statistics engine."""
        self.percentile_thresholds = config.percentile_thresholds
        
    def compute_session_statistics(self, 
                                  session_data: pd.DataFrame,
                                  price_column: str = 'close') -> Dict[str, float]:
        """Compute comprehensive statistics for a trading session."""
        prices = session_data[price_column].values
        
        mean = float(np.mean(prices))
        std = float(np.std(prices, ddof=1))
        skewness = float(stats.skew(prices))
        kurtosis = float(stats.kurtosis(prices, fisher=True))
        
        # Percentiles
        percentiles = {}
        for p in self.percentile_thresholds:
            key = f'p{int(p)}'
            percentiles[key] = float(np.percentile(prices, p))
        
        # Returns statistics
        returns = session_data['returns'].dropna().values
        returns_stats = {}
        if len(returns) > 0:
            returns_stats = {
                'returns_mean': float(np.mean(returns)),
                'returns_std': float(np.std(returns, ddof=1)),
                'returns_skew': float(stats.skew(returns)),
                'returns_kurtosis': float(stats.kurtosis(returns, fisher=True)),
                'returns_sharpe': float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            }
        
        stats_dict = {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min': float(np.min(prices)),
            'max': float(np.max(prices)),
            **percentiles,
            **returns_stats
        }
        
        # Add statistical levels
        for sigma in [1, 2, 3]:
            stats_dict[f'plus_{sigma}sigma'] = float(mean + sigma * std)
            stats_dict[f'minus_{sigma}sigma'] = float(mean - sigma * std)
        
        return stats_dict


# ============================================================================
# DISTRIBUTION MODELS
# ============================================================================

class DistributionModels:
    """Fits and compares different distribution models."""
    
    def fit_all_distributions(self, data: np.ndarray, is_returns: bool = False) -> Dict[str, Any]:
        """Fit all distributions and compute goodness-of-fit metrics."""
        results = {}
        n_obs = len(data)
        
        if n_obs < 10:
            logger.warning("Insufficient data for distribution fitting")
            return results
        
        # Normal distribution
        try:
            norm_params = stats.norm.fit(data)
            norm_ll = np.sum(stats.norm.logpdf(data, *norm_params))
            results['normal'] = {
                'params': {'loc': float(norm_params[0]), 'scale': float(norm_params[1])},
                'log_likelihood': float(norm_ll),
                'aic': float(2 * 2 - 2 * norm_ll),
                'bic': float(2 * np.log(n_obs) - 2 * norm_ll)
            }
        except Exception as e:
            logger.error(f"Error fitting normal: {e}")
            results['normal'] = None
        
        # Student-t distribution
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_params = stats.t.fit(data)
                t_ll = np.sum(stats.t.logpdf(data, *t_params))
                results['student_t'] = {
                    'params': {'df': float(t_params[0]), 'loc': float(t_params[1]), 'scale': float(t_params[2])},
                    'log_likelihood': float(t_ll),
                    'aic': float(2 * 3 - 2 * t_ll),
                    'bic': float(3 * np.log(n_obs) - 2 * t_ll)
                }
        except Exception as e:
            logger.error(f"Error fitting Student-t: {e}")
            results['student_t'] = None
        
        # KDE
        try:
            kde = gaussian_kde(data)
            kde_ll = np.sum(np.log(kde.evaluate(data) + 1e-10))
            results['kde'] = {
                'params': {'bw_method': 'scott'},
                'log_likelihood': float(kde_ll),
                'aic': float(2 * 5 - 2 * kde_ll),
                'bic': float(5 * np.log(n_obs) - 2 * kde_ll),
                'kde_object': kde
            }
        except Exception as e:
            logger.error(f"Error fitting KDE: {e}")
            results['kde'] = None
        
        return results


# ============================================================================
# FAT TAIL ANALYSIS
# ============================================================================

class FatTailAnalyzer:
    """Analyzes fat-tail characteristics."""
    
    def compute_tail_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive tail risk metrics."""
        metrics = {}
        
        if len(returns) < 10:
            return metrics
        
        # Excess kurtosis
        metrics['excess_kurtosis'] = float(stats.kurtosis(returns, fisher=True))
        
        # Tail indices using Hill estimator
        abs_returns = np.abs(returns)
        sorted_returns = np.sort(abs_returns)[::-1]
        
        for fraction in [0.01, 0.025, 0.05]:
            k = int(len(sorted_returns) * fraction)
            if k >= 5:
                tail_data = sorted_returns[:k]
                hill = 1.0 / np.mean(np.log(tail_data / tail_data[-1] + 1e-10))
                metrics[f'tail_index_{fraction:.3f}'] = float(hill)
        
        # Extreme probabilities
        std_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        for t in [2.0, 3.0, 4.0, 5.0]:
            metrics[f'p_abs_z>{t}'] = float(np.mean(np.abs(std_returns) > t))
            metrics[f'p_z>{t}'] = float(np.mean(std_returns > t))
            metrics[f'p_z<-{t}'] = float(np.mean(std_returns < -t))
        
        # VaR and ES
        for conf in [0.95, 0.99]:
            var = np.percentile(returns, (1 - conf) * 100)
            es = np.mean(returns[returns <= var]) if len(returns[returns <= var]) > 0 else var
            metrics[f'var_{conf:.3f}'] = float(var)
            metrics[f'es_{conf:.3f}'] = float(es)
        
        return metrics


# ============================================================================
# EXTREME DETECTOR
# ============================================================================

class ExtremeDetector:
    """Detects extreme price movements."""
    
    def detect_extremes(self, prices: np.ndarray, mean: float, std: float) -> Dict[str, np.ndarray]:
        """Detect extreme observations based on z-scores."""
        if std == 0:
            return {}
        
        z_scores = (prices - mean) / std
        extremes = {}
        
        for t in [2.0, 3.0]:
            extremes[f'z>{t}'] = z_scores > t
            extremes[f'z<-{t}'] = z_scores < -t
        
        return extremes


# ============================================================================
# VOLATILITY REGIME
# ============================================================================

class VolatilityRegimeDetector:
    """Detects volatility regimes."""
    
    def __init__(self):
        self.rolling_window = 20
    
    def analyze_regimes(self, session_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility regimes in the session."""
        if 'returns' not in session_data.columns:
            returns = session_data['close'].pct_change().dropna()
        else:
            returns = session_data['returns'].dropna()
        
        if len(returns) < self.rolling_window:
            return {}
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.rolling_window).std()
        rolling_vol = rolling_vol * np.sqrt(252 * 390)  # Annualized
        
        # Define regime thresholds
        low_thresh = rolling_vol.quantile(0.25)
        high_thresh = rolling_vol.quantile(0.75)
        extreme_thresh = rolling_vol.quantile(0.95)
        
        # Calculate regime proportions
        summary = {
            'avg_volatility': float(rolling_vol.mean()),
            'max_volatility': float(rolling_vol.max()),
            'low_proportion': float((rolling_vol <= low_thresh).mean()),
            'normal_proportion': float(((rolling_vol > low_thresh) & (rolling_vol <= high_thresh)).mean()),
            'high_proportion': float(((rolling_vol > high_thresh) & (rolling_vol <= extreme_thresh)).mean()),
            'extreme_proportion': float((rolling_vol > extreme_thresh).mean())
        }
        
        return summary


# ============================================================================
# LEVEL STORE
# ============================================================================

class LevelStore:
    """Stores and retrieves static session levels."""
    
    def __init__(self):
        """Initialize the level store."""
        self.store_dir = Path("./data/levels")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.levels: Dict[str, Dict[str, Any]] = {}
        self._load_store()
        
    def _load_store(self):
        """Load existing level store from disk."""
        store_file = self.store_dir / "session_levels.pkl"
        if store_file.exists():
            try:
                with open(store_file, 'rb') as f:
                    self.levels = pickle.load(f)
                logger.info(f"Loaded {len(self.levels)} sessions from level store")
            except Exception as e:
                logger.error(f"Error loading level store: {e}")
    
    def _save_store(self):
        """Save level store to disk."""
        store_file = self.store_dir / "session_levels.pkl"
        try:
            with open(store_file, 'wb') as f:
                pickle.dump(self.levels, f)
        except Exception as e:
            logger.error(f"Error saving level store: {e}")
    
    def save_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Save session data."""
        try:
            self.levels[session_id] = data
            self._save_store()
            logger.info(f"Saved session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return self.levels.get(session_id)
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        return sorted(self.levels.keys())


# ============================================================================
# ADVANCED DASHBOARD
# ============================================================================

# Custom CSS for professional look
CUSTOM_CSS = """
:root {
    --primary-bg: #1e1e2f;
    --secondary-bg: #2d2d44;
    --card-bg: #2a2a3c;
    --accent-1: #7289da;
    --accent-2: #43b581;
    --accent-3: #f04747;
    --accent-4: #faa61a;
    --text-primary: #ffffff;
    --text-secondary: #b9bbbe;
    --border-color: #40404e;
}

body {
    background-color: var(--primary-bg);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.nav-tabs {
    border-bottom: 2px solid var(--border-color);
    margin-bottom: 20px;
}

.nav-tabs .nav-link {
    color: var(--text-secondary);
    border: none;
    padding: 12px 20px;
    font-weight: 500;
    transition: all 0.2s;
}

.nav-tabs .nav-link:hover {
    color: var(--text-primary);
    background: rgba(114, 137, 218, 0.1);
    border: none;
}

.nav-tabs .nav-link.active {
    color: var(--accent-1);
    background: transparent;
    border-bottom: 3px solid var(--accent-1);
}

.card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    margin-bottom: 20px;
    transition: transform 0.2s, box-shadow 0.2s;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.3);
}

.card-header {
    background: rgba(0,0,0,0.2);
    border-bottom: 2px solid var(--border-color);
    color: var(--text-primary);
    font-weight: 600;
    font-size: 1.1rem;
    padding: 15px 20px;
    border-radius: 12px 12px 0 0 !important;
}

.card-body {
    padding: 20px;
    color: var(--text-secondary);
}

.btn-primary {
    background: var(--accent-1);
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    transition: all 0.2s;
}

.btn-primary:hover {
    background: #5b6eae;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(114, 137, 218, 0.3);
}

.metric-card {
    background: var(--secondary-bg);
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    color: var(--text-primary);
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 5px;
}

.metric-change {
    font-size: 0.9rem;
    margin-top: 5px;
}

.change-positive {
    color: var(--accent-2);
}

.change-negative {
    color: var(--accent-3);
}

.progress-bar-custom {
    height: 8px;
    border-radius: 4px;
    margin: 10px 0;
}

.table-custom {
    background: transparent;
    color: var(--text-secondary);
}

.table-custom th {
    border-bottom: 2px solid var(--border-color);
    color: var(--text-primary);
    font-weight: 600;
}

.table-custom td {
    border-bottom: 1px solid var(--border-color);
}

.tooltip-custom {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px;
    color: var(--text-primary);
}

.header-gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 0 0 30px 30px;
    margin-bottom: 30px;
}

.footer-gradient {
    background: linear-gradient(135deg, #2d2d44 0%, #1a1a2e 100%);
    padding: 20px;
    border-radius: 30px 30px 0 0;
    margin-top: 30px;
}

.session-badge {
    background: var(--accent-1);
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-left: 10px;
}

.complete-badge {
    background: var(--accent-2);
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-left: 10px;
}

.incomplete-badge {
    background: var(--accent-3);
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-left: 10px;
}
"""


class ESDashboard:
    """Advanced interactive dashboard for ES futures analysis."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[
                dbc.themes.DARKLY,
                "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
            ],
            title="ES Futures Research Platform",
            suppress_callback_exceptions=True
        )
        server = self.app.server
        
        # Add custom CSS
        self.app.index_string = self.app.index_string.replace(
            '</head>',
            f'<style>{CUSTOM_CSS}</style></head>'
        )
        
        self.data_fetcher = DataFetcher()
        self.session_manager = SessionManager()
        self.statistics_engine = StatisticsEngine()
        self.distribution_models = DistributionModels()
        self.tail_analyzer = FatTailAnalyzer()
        self.extreme_detector = ExtremeDetector()
        self.regime_detector = VolatilityRegimeDetector()
        self.hmm_detector = HMMRegimeDetector()
        self.level_store = LevelStore()
        
        self.data = None
        self.sessions = {}
        self.hmm_results = {}
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup the advanced dashboard layout with multiple tabs."""
        
        self.app.layout = dbc.Container([
            # Header with gradient
            html.Div([
                html.Div([
                    html.H1("ES Futures Quantitative Research Platform",
                           className="text-center mb-2",
                           style={'color': 'white', 'fontWeight': '700'}),
                    html.H5("Complete CME Sessions: 18:00 → 17:00 Next Day (23 hours)",
                           className="text-center mb-4",
                           style={'color': 'rgba(255,255,255,0.9)'}),
                    
                    # Quick metrics row
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div("Current Price", className="metric-label"),
                                html.Div(id="current-price", className="metric-value"),
                                html.Div(id="price-change", className="metric-change")
                            ], className="metric-card")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.Div("Session Vol", className="metric-label"),
                                html.Div(id="session-vol", className="metric-value"),
                                html.Div(id="vol-regime", className="metric-change")
                            ], className="metric-card")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.Div("HMM Regime", className="metric-label"),
                                html.Div(id="hmm-regime", className="metric-value"),
                                html.Div(id="hmm-prob", className="metric-change")
                            ], className="metric-card")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.Div("Extreme Events", className="metric-label"),
                                html.Div(id="extreme-count", className="metric-value"),
                                html.Div(id="extreme-latest", className="metric-change")
                            ], className="metric-card")
                        ], width=3)
                    ], className="mt-4")
                ], className="header-gradient")
            ]),
            
            # Main tabs
            dbc.Tabs([
                # Main Analysis Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.I(className="fas fa-chart-line mr-2"),
                                    "Session Controls"
                                ]),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Select Session:", 
                                                      style={'color': 'var(--text-primary)'}),
                                            dcc.Dropdown(
                                                id='session-dropdown',
                                                placeholder='Select a trading session',
                                                style={'color': 'black'}
                                            )
                                        ], width=4),
                                        dbc.Col([
                                            html.Label("Analysis Type:", 
                                                      style={'color': 'var(--text-primary)'}),
                                            dcc.RadioItems(
                                                id='analysis-type',
                                                options=[
                                                    {'label': ' Price Distribution', 'value': 'price'},
                                                    {'label': ' Returns Distribution', 'value': 'returns'}
                                                ],
                                                value='price',
                                                inline=True,
                                                labelStyle={'color': 'var(--text-secondary)', 'marginRight': '20px'}
                                            )
                                        ], width=4),
                                        dbc.Col([
                                            html.Label("Show Levels:", 
                                                      style={'color': 'var(--text-primary)'}),
                                            dcc.Checklist(
                                                id='show-levels',
                                                options=[
                                                    {'label': ' Mean', 'value': 'mean'},
                                                    {'label': ' ±1σ', 'value': 'sigma1'},
                                                    {'label': ' ±2σ', 'value': 'sigma2'},
                                                    {'label': ' ±3σ', 'value': 'sigma3'},
                                                    {'label': ' %', 'value': 'percentiles'}
                                                ],
                                                value=['mean', 'sigma2'],
                                                inline=True,
                                                labelStyle={'color': 'var(--text-secondary)', 'marginRight': '15px'}
                                            )
                                        ], width=4)
                                    ]),
                                    html.Div(id="session-range-info", className="mt-3")
                                ])
                            ])
                        ])
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.I(className="fas fa-candlestick-chart mr-2"),
                                    "Price Action with Statistical Levels (Full Session)"
                                ]),
                                dbc.CardBody([
                                    dcc.Graph(id='price-chart', style={'height': '500px'})
                                ])
                            ])
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.I(className="fas fa-chart-bar mr-2"),
                                    "Distribution Analysis"
                                ]),
                                dbc.CardBody([
                                    dcc.Graph(id='dist-chart', style={'height': '500px'})
                                ])
                            ])
                        ], width=4)
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Basic Statistics"),
                                dbc.CardBody(id='stats-panel')
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Tail Analysis"),
                                dbc.CardBody(id='tail-panel')
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Distribution Fit"),
                                dbc.CardBody(id='dist-panel')
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Volatility Regime"),
                                dbc.CardBody(id='regime-panel')
                            ])
                        ], width=3)
                    ], className="mb-4")
                ], label="Main Analysis", tab_id="tab-main"),
                
                # HMM Regimes Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.I(className="fas fa-project-diagram mr-2"),
                                    "Hidden Markov Model Regimes"
                                ]),
                                dbc.CardBody([
                                    html.Div(id="hmm-status"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.H6("State Parameters", 
                                                   style={'color': 'var(--text-primary)'}),
                                            html.Div(id="hmm-state-params")
                                        ], width=6),
                                        dbc.Col([
                                            html.H6("Transition Matrix", 
                                                   style={'color': 'var(--text-primary)'}),
                                            dcc.Graph(id="hmm-transition-matrix")
                                        ], width=6)
                                    ]),
                                    html.Hr(),
                                    html.H6("Regime Evolution", 
                                           style={'color': 'var(--text-primary)'}),
                                    dcc.Graph(id="hmm-states-chart", style={'height': '400px'})
                                ])
                            ])
                        ])
                    ])
                ], label="HMM Regimes", tab_id="tab-hmm"),
                
                # Volume Profile Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Volume Profile"),
                                dbc.CardBody([
                                    dcc.Graph(id="volume-profile-chart")
                                ])
                            ])
                        ])
                    ])
                ], label="Volume Profile", tab_id="tab-volume"),
                
                # Extreme Events Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Extreme Events Detection"),
                                dbc.CardBody([
                                    html.Div(id="extreme-events-table")
                                ])
                            ])
                        ])
                    ])
                ], label="Extreme Events", tab_id="tab-extreme"),
                
                # Settings Tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Configuration"),
                                dbc.CardBody([
                                    html.H6("Data Settings"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Resolution:"),
                                            dcc.Dropdown(
                                                id='settings-resolution',
                                                options=[
                                                    {'label': '1 Minute', 'value': '1m'},
                                                    {'label': '5 Minutes', 'value': '5m'}
                                                ],
                                                value=config.data_resolution,
                                                style={'color': 'black'}
                                            )
                                        ], width=4),
                                        dbc.Col([
                                            html.Label("Days of History:"),
                                            dcc.Input(
                                                id='settings-days',
                                                type='number',
                                                value=60,
                                                min=5,
                                                max=90,
                                                className="form-control"
                                            )
                                        ], width=4),
                                        dbc.Col([
                                            html.Label("HMM States:"),
                                            dcc.Input(
                                                id='settings-hmm-states',
                                                type='number',
                                                value=config.hmm_n_states,
                                                min=2,
                                                max=6,
                                                className="form-control"
                                            )
                                        ], width=4)
                                    ]),
                                    html.Hr(),
                                    dbc.Button("Refresh Data", id="refresh-button", 
                                              color="primary", className="mt-3")
                                ])
                            ])
                        ])
                    ])
                ], label="Settings", tab_id="tab-settings")
            ], id="main-tabs", active_tab="tab-main", className="mt-3"),
            
            # Footer
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.P("© 2026 ES Futures Research Platform | Complete CME Sessions 18:00 → 17:00 | Levels static per session",
                              className="text-center small",
                              style={'color': 'var(--text-secondary)'})
                    ])
                ])
            ], className="footer-gradient mt-4")
            
        ], fluid=True, style={'backgroundColor': 'var(--primary-bg)', 'minHeight': '100vh'})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('session-dropdown', 'options'),
             Output('session-dropdown', 'value'),
             Output('session-range-info', 'children')],
            [Input('session-dropdown', 'id')]
        )
        def update_session_list(_):
            session_ids = self.level_store.get_all_sessions()
            options = [{'label': sid, 'value': sid} for sid in session_ids]
            value = session_ids[-1] if session_ids else None
            
            # Get current session info
            if value and value in self.sessions:
                session_data = self.sessions[value]
                start = session_data.index[0]
                end = session_data.index[-1]
                
                # Get expected session range
                session_date = datetime.strptime(value, '%Y-%m-%d')
                session_date = pytz.timezone(config.timezone).localize(session_date)
                expected_start, expected_end = get_full_session_range(session_date)
                
                # Check if session is complete (allow 1 hour margin)
                start_diff = abs((start - expected_start).total_seconds())
                end_diff = abs((end - expected_end).total_seconds())
                
                if start_diff < 3600 and end_diff < 3600:
                    status_badge = html.Span("✅ COMPLETE", className="complete-badge")
                else:
                    status_badge = html.Span("⚠️ INCOMPLETE", className="incomplete-badge")
                
                range_info = html.Div([
                    html.Span(f"Full Session: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} "),
                    html.Span(f"({len(session_data)} candles) "),
                    status_badge
                ])
            else:
                range_info = "No session selected"
            
            return options, value, range_info
        
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('dist-chart', 'figure'),
             Output('stats-panel', 'children'),
             Output('tail-panel', 'children'),
             Output('dist-panel', 'children'),
             Output('regime-panel', 'children'),
             Output('current-price', 'children'),
             Output('price-change', 'children'),
             Output('session-vol', 'children'),
             Output('vol-regime', 'children'),
             Output('extreme-count', 'children'),
             Output('extreme-latest', 'children'),
             Output('hmm-regime', 'children'),
             Output('hmm-prob', 'children'),
             Output('hmm-state-params', 'children'),
             Output('hmm-transition-matrix', 'figure'),
             Output('hmm-states-chart', 'figure'),
             Output('extreme-events-table', 'children')],
            [Input('session-dropdown', 'value'),
             Input('analysis-type', 'value'),
             Input('show-levels', 'value')]
        )
        def update_dashboard(session_id, analysis_type, show_levels):


        # GUARD - data ještě nejsou načtená
            empty_fig = go.Figure()
            empty_fig.update_layout(
                template='darkly',
                title={'text': 'Načítám data, čekejte...'}
            )
            no_data = html.Span("načítám...", className="text-warning")
            
            if not _dashboard.sessions:
                # Vrať přesně 18 prázdných hodnot
                return (
                    empty_fig,  # price-chart
                    empty_fig,  # dist-chart
                    no_data,    # stats-panel
                    no_data,    # tail-panel
                    no_data,    # dist-panel
                    no_data,    # regime-panel
                    "--",       # current-price
                    no_data,    # price-change
                    "--",       # session-vol
                    "--",       # vol-regime
                    "--",       # extreme-count
                    "--",       # extreme-latest
                    "--",       # hmm-regime
                    "--",       # hmm-prob
                    no_data,    # hmm-state-params
                    empty_fig,  # hmm-transition
                    empty_fig,  # hmm-states-chart
                    no_data,    # extreme-events-table
                )


            if not session_id or session_id not in self.sessions:
                empty_fig = go.Figure()
                empty_fig.update_layout(template="plotly_dark", title="No data")
                empty_panel = html.P("No data", className="text-warning")
                empty_metrics = ["--"] * 4
                empty_hmm = html.P("No HMM data", className="text-warning")
                empty_fig2 = go.Figure()
                empty_table = html.P("No extreme events", className="text-warning")
                return ([empty_fig] * 2 + [empty_panel] * 4 + 
                        empty_metrics * 2 + [empty_panel] * 2 + 
                        [empty_hmm, empty_fig2, empty_fig2, empty_table])
            
            session_data = self.sessions[session_id]
            session_info = self.level_store.get_session(session_id)
            
            if not session_info:
                return [go.Figure()] * 2 + [html.P("Error")] * 10 + [go.Figure()] * 2 + [html.P("Error")]
            
            # Get HMM results
            if session_id not in self.hmm_results:
                returns = session_data['returns'].dropna().values
                self.hmm_results[session_id] = self.hmm_detector.fit_hmm(returns, session_id)
            
            hmm_result = self.hmm_results.get(session_id, {})
            current_hmm = self.hmm_detector.get_current_regime(
                session_id, session_data['returns'].dropna().values[-50:]
            ) if session_id in self.hmm_detector.models else {}
            
            # Create charts
            price_fig = self._create_price_chart(
                session_data, session_info, show_levels
            )
            
            dist_fig = self._create_dist_chart(
                session_data, session_info, analysis_type
            )
            
            # Create panels
            stats_panel = self._create_stats_panel(session_info)
            tail_panel = self._create_tail_panel(session_info)
            dist_panel = self._create_dist_fit_panel(session_info)
            regime_panel = self._create_regime_panel(session_info)
            
            # Current metrics
            current_price = f"${session_data['close'].iloc[-1]:.2f}"
            price_change_pct = session_data['returns'].iloc[-1] * 100
            price_change = html.Span(
                f"{price_change_pct:+.2f}%",
                className="change-positive" if price_change_pct > 0 else "change-negative"
            )
            
            session_vol = f"{session_info.get('regime_summary', {}).get('avg_volatility', 0):.2f}%"
            vol_regime = self._get_current_regime(session_data, session_info)
            
            # Extreme events
            extremes = self._detect_extremes(session_data, session_info)
            extreme_count = str(len(extremes))
            extreme_latest = extremes.iloc[0]['time'].strftime('%H:%M') if not extremes.empty else "--"
            
            # HMM metrics
            hmm_regime = current_hmm.get('current_regime', '--')
            hmm_prob = f"{max(current_hmm.get('regime_probs', [0]))*100:.1f}%" if current_hmm else "--"
            
            # HMM panels
            hmm_params = self._create_hmm_params_panel(hmm_result)
            hmm_transition = self.hmm_detector.visualize_transitions(session_id) if session_id in self.hmm_detector.models else go.Figure()
            hmm_states = self._create_hmm_states_chart(session_data, hmm_result) if hmm_result else go.Figure()
            
            # Extreme events table
            extreme_table = self._create_extreme_table(extremes)
            
            return (price_fig, dist_fig, stats_panel, tail_panel, dist_panel, regime_panel,
                    current_price, price_change, session_vol, vol_regime,
                    extreme_count, extreme_latest, hmm_regime, hmm_prob,
                    hmm_params, hmm_transition, hmm_states, extreme_table)
    
    def _create_price_chart(self, session_data: pd.DataFrame, 
                           session_info: dict, show_levels: list) -> go.Figure:
        """Create enhanced price chart with levels showing FULL session - NO VERTICAL LINES."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price Action - Full Session (18:00 → 17:00)", "Volume")
        )
        
        # Candlestick - simplified hover
        fig.add_trace(
            go.Candlestick(
                x=session_data.index,
                open=session_data['open'],
                high=session_data['high'],
                low=session_data['low'],
                close=session_data['close'],
                name='ES',
                showlegend=False,
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                hoverinfo='none'  # Disable hover
            ),
            row=1, col=1
        )
        
        # Volume with colors - simplified hover
        colors = ['#26a69a' if session_data['close'].iloc[i] >= session_data['open'].iloc[i] 
                 else '#ef5350' for i in range(len(session_data))]
        fig.add_trace(
            go.Bar(
                x=session_data.index,
                y=session_data['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False,
                hoverinfo='none'  # Disable hover
            ),
            row=2, col=1
        )
        
        # Add statistical levels
        levels = session_info.get('levels', {})
        
        if 'mean' in show_levels and 'mean' in levels:
            fig.add_hline(y=levels['mean'], line_color='white', 
                         line_dash='solid', opacity=0.5, row=1, col=1,
                         annotation_text=f"Mean: ${levels['mean']:.2f}", annotation_position="right")
        
        if 'sigma1' in show_levels:
            if 'plus_1sigma' in levels:
                fig.add_hline(y=levels['plus_1sigma'], line_color='#ffd700', 
                             line_dash='dash', opacity=0.3, row=1, col=1,
                             annotation_text=f"+1σ: ${levels['plus_1sigma']:.2f}", annotation_position="right")
            if 'minus_1sigma' in levels:
                fig.add_hline(y=levels['minus_1sigma'], line_color='#ffd700', 
                             line_dash='dash', opacity=0.3, row=1, col=1,
                             annotation_text=f"-1σ: ${levels['minus_1sigma']:.2f}", annotation_position="left")
        
        if 'sigma2' in show_levels:
            if 'plus_2sigma' in levels:
                fig.add_hline(y=levels['plus_2sigma'], line_color='#ffa500', 
                             line_dash='dash', opacity=0.3, row=1, col=1,
                             annotation_text=f"+2σ: ${levels['plus_2sigma']:.2f}", annotation_position="right")
            if 'minus_2sigma' in levels:
                fig.add_hline(y=levels['minus_2sigma'], line_color='#ffa500', 
                             line_dash='dash', opacity=0.3, row=1, col=1,
                             annotation_text=f"-2σ: ${levels['minus_2sigma']:.2f}", annotation_position="left")
        
        if 'sigma3' in show_levels:
            if 'plus_3sigma' in levels:
                fig.add_hline(y=levels['plus_3sigma'], line_color='#ff4500', 
                             line_dash='dash', opacity=0.3, row=1, col=1,
                             annotation_text=f"+3σ: ${levels['plus_3sigma']:.2f}", annotation_position="right")
            if 'minus_3sigma' in levels:
                fig.add_hline(y=levels['minus_3sigma'], line_color='#ff4500', 
                             line_dash='dash', opacity=0.3, row=1, col=1,
                             annotation_text=f"-3σ: ${levels['minus_3sigma']:.2f}", annotation_position="left")
        
        if 'percentiles' in show_levels:
            for p, color in [(1, '#9370db'), (5, '#9370db'), (95, '#9370db'), (99, '#9370db')]:
                key = f'p{p}'
                if key in levels:
                    fig.add_hline(y=levels[key], line_color=color, 
                                 line_dash='dot', opacity=0.2, row=1, col=1,
                                 annotation_text=f"{p}th: ${levels[key]:.2f}", 
                                 annotation_position="right" if p > 50 else "left")
        
        fig.update_layout(
            template="plotly_dark",
            title=f"Complete Session {session_data.index[0].strftime('%Y-%m-%d')} (18:00 to 17:00)",
            hovermode=False,  # Disable all hover interactions
            height=600,
            margin=dict(l=50, r=150, t=80, b=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)'
            )
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_traces(hoverinfo='none')  # Additional hover disabling
        return fig
    
    def _create_dist_chart(self, session_data: pd.DataFrame, 
                          session_info: dict, analysis_type: str) -> go.Figure:
        """Create enhanced distribution chart."""
        fig = go.Figure()
        
        if analysis_type == 'price':
            data = session_data['close'].values
            xlabel = "Price ($)"
            title = "Price Distribution (Full Session)"
        else:
            data = session_data['returns'].dropna().values * 100
            xlabel = "Returns (%)"
            title = "Returns Distribution (Full Session)"
        
        # Histogram with custom styling - simplified hover
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=50,
            name='Observed',
            opacity=0.7,
            histnorm='probability density',
            marker_color='#4a90e2',
            marker_line_color='#2c3e50',
            marker_line_width=1,
            hoverinfo='none'  # Disable hover
        ))
        
        # Fitted distributions
        dist_models = session_info.get('distribution_models', {})
        
        if analysis_type == 'returns' and 'returns_distributions' in session_info:
            dist_models = session_info['returns_distributions']
        
        x_range = np.linspace(np.min(data), np.max(data), 200)
        
        if dist_models.get('normal'):
            params = dist_models['normal']['params'].copy()
            if analysis_type == 'returns':
                params['scale'] = params['scale'] * 100
            y_norm = stats.norm.pdf(x_range, **params)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_norm, name='Normal',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                hoverinfo='none'
            ))
        
        if dist_models.get('student_t'):
            params = dist_models['student_t']['params'].copy()
            if analysis_type == 'returns':
                params['scale'] = params['scale'] * 100
            y_t = stats.t.pdf(x_range, **params)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_t, name='Student-t',
                line=dict(color='#51cf66', width=2, dash='dot'),
                hoverinfo='none'
            ))
        
        if dist_models.get('kde'):
            if analysis_type == 'returns' and 'returns_distributions' in session_info:
                kde_data = data / 100
                kde = gaussian_kde(kde_data)
                y_kde = kde.evaluate(x_range/100) / 100
            else:
                if 'kde_object' in dist_models['kde']:
                    kde = dist_models['kde']['kde_object']
                    y_kde = kde.evaluate(x_range)
                else:
                    kde = gaussian_kde(data)
                    y_kde = kde.evaluate(x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range, y=y_kde, name='KDE',
                line=dict(color='#ffd700', width=2),
                hoverinfo='none'
            ))
        
        fig.update_layout(
            template="plotly_dark",
            title=title,
            xaxis_title=xlabel,
            yaxis_title="Probability Density",
            hovermode=False,
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            bargap=0.05
        )
        
        fig.update_traces(hoverinfo='none')
        return fig
    
    def _create_stats_panel(self, session_info: dict) -> html.Div:
        """Create enhanced statistics panel with proper percentiles."""
        stats = session_info.get('statistics', {})
        if not stats:
            return html.P("No statistics", className="text-warning")
        
        items = []
        
        # Price stats with icons
        price_items = [
            ('💰 Mean', 'mean', '${:.2f}'),
            ('📊 Std Dev', 'std', '${:.2f}'),
            ('📈 Skewness', 'skewness', '{:.4f}'),
            ('📉 Kurtosis', 'kurtosis', '{:.4f}'),
            ('🔻 Min', 'min', '${:.2f}'),
            ('🔺 Max', 'max', '${:.2f}')
        ]
        
        for label, key, fmt in price_items:
            if key in stats:
                items.append(html.Div([
                    html.Span(label, style={'color': '#aaa'}),
                    html.Span(fmt.format(stats[key]), 
                             style={'color': '#fff', 'float': 'right'})
                ], className="mb-2"))
        
        # Percentiles
        items.append(html.Div("📌 Percentiles:", 
                             style={'color': '#ffd43b', 'marginTop': '15px', 'fontWeight': 'bold'}))
        for p in [1, 5, 95, 99]:
            key = f'p{p}'
            if key in stats:
                items.append(html.Div([
                    html.Span(f"  {p}th", style={'color': '#ccc'}),
                    html.Span(f"${stats[key]:.2f}", 
                             style={'color': '#9370db', 'float': 'right'})
                ], className="mb-1"))
        
        # Return stats
        if 'returns_mean' in stats:
            items.append(html.Div("📈 Return Stats:", 
                                 style={'color': '#ffd43b', 'marginTop': '15px', 'fontWeight': 'bold'}))
            items.append(html.Div([
                html.Span("  Mean", style={'color': '#ccc'}),
                html.Span(f"{stats['returns_mean']*100:.4f}%", 
                         style={'color': '#69db7e', 'float': 'right'})
            ], className="mb-1"))
            items.append(html.Div([
                html.Span("  Std Dev", style={'color': '#ccc'}),
                html.Span(f"{stats['returns_std']*100:.4f}%", 
                         style={'color': '#69db7e', 'float': 'right'})
            ], className="mb-1"))
            if 'returns_sharpe' in stats:
                items.append(html.Div([
                    html.Span("  Sharpe", style={'color': '#ccc'}),
                    html.Span(f"{stats['returns_sharpe']:.2f}", 
                             style={'color': '#ffa94d', 'float': 'right'})
                ], className="mb-1"))
        
        return html.Div(items, style={'fontFamily': 'monospace', 'fontSize': '14px'})
    
    def _create_tail_panel(self, session_info: dict) -> html.Div:
        """Create enhanced tail analysis panel."""
        tail = session_info.get('tail_metrics', {})
        if not tail:
            return html.P("No tail analysis", className="text-warning")
        
        items = []
        
        # Excess kurtosis
        if 'excess_kurtosis' in tail:
            kurt = tail['excess_kurtosis']
            color = '#ff6b6b' if kurt > 3 else '#69db7e'
            icon = '⚠️' if kurt > 3 else '✅'
            items.append(html.Div([
                html.Span(f"{icon} Excess Kurtosis: ", style={'color': '#aaa'}),
                html.Span(f"{kurt:.4f}", style={'color': color, 'float': 'right'})
            ], className="mb-2"))
        
        # Tail indices
        items.append(html.Div("📊 Tail Indices:", 
                             style={'color': '#ffd43b', 'marginTop': '15px', 'fontWeight': 'bold'}))
        for frac in [0.01, 0.025, 0.05]:
            key = f'tail_index_{frac:.3f}'
            if key in tail:
                items.append(html.Div([
                    html.Span(f"  α ({frac*100:.1f}%): ", style={'color': '#ccc'}),
                    html.Span(f"{tail[key]:.2f}", 
                             style={'color': '#51cf66', 'float': 'right'})
                ], className="mb-1"))
        
        # Extreme probabilities
        items.append(html.Div("🎯 Extreme Probs:", 
                             style={'color': '#ffd43b', 'marginTop': '15px', 'fontWeight': 'bold'}))
        for z in [2.0, 3.0, 4.0]:
            key = f'p_abs_z>{z}'
            if key in tail:
                prob = tail[key] * 100
                color = '#ff6b6b' if prob > 5 else '#69db7e'
                items.append(html.Div([
                    html.Span(f"  P(|Z|>{z}): ", style={'color': '#ccc'}),
                    html.Span(f"{prob:.2f}%", style={'color': color, 'float': 'right'})
                ], className="mb-1"))
        
        # VaR
        items.append(html.Div("🛡️ Risk Metrics:", 
                             style={'color': '#ffd43b', 'marginTop': '15px', 'fontWeight': 'bold'}))
        for conf in [0.95, 0.99]:
            key = f'var_{conf:.3f}'
            if key in tail:
                var = tail[key] * 100
                items.append(html.Div([
                    html.Span(f"  VaR {conf:.0%}: ", style={'color': '#ccc'}),
                    html.Span(f"{var:.4f}%", style={'color': '#ffa94d', 'float': 'right'})
                ], className="mb-1"))
        
        return html.Div(items, style={'fontFamily': 'monospace', 'fontSize': '13px'})
    
    def _create_dist_fit_panel(self, session_info: dict) -> html.Div:
        """Create enhanced distribution fit panel."""
        dist_models = session_info.get('distribution_models', {})
        if not dist_models:
            return html.P("No fits", className="text-warning")
        
        items = []
        
        # Find best model by AIC
        best_aic = float('inf')
        best_model = None
        for name, data in dist_models.items():
            if data and 'aic' in data:
                if data['aic'] < best_aic:
                    best_aic = data['aic']
                    best_model = name
        
        model_icons = {
            'normal': '📊',
            'student_t': '📈',
            'kde': '📉'
        }
        
        for name in ['normal', 'student_t', 'kde']:
            if name in dist_models and dist_models[name]:
                data = dist_models[name]
                display_name = {'normal': 'Normal', 'student_t': 'Student-t', 'kde': 'KDE'}[name]
                is_best = (name == best_model)
                
                items.append(html.Div([
                    html.Div(f"{model_icons.get(name, '📊')} {display_name}", style={
                        'color': '#51cf66' if is_best else '#ffd43b',
                        'fontWeight': 'bold',
                        'marginTop': '12px',
                        'fontSize': '14px'
                    }),
                    html.Div([
                        html.Span("AIC: ", style={'color': '#aaa'}),
                        html.Span(f"{data['aic']:.2f}", 
                                 style={'color': '#fff', 'float': 'right'})
                    ], className="mb-1"),
                    html.Div([
                        html.Span("BIC: ", style={'color': '#aaa'}),
                        html.Span(f"{data['bic']:.2f}", 
                                 style={'color': '#fff', 'float': 'right'})
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Log-Likelihood: ", style={'color': '#aaa'}),
                        html.Span(f"{data['log_likelihood']:.2f}", 
                                 style={'color': '#69db7e', 'float': 'right'})
                    ], className="mb-1")
                ]))
        
        return html.Div(items, style={'fontFamily': 'monospace', 'fontSize': '13px'})
    
    def _create_regime_panel(self, session_info: dict) -> html.Div:
        """Create enhanced volatility regime panel."""
        regime = session_info.get('regime_summary', {})
        if not regime:
            return html.P("No regime data", className="text-warning")
        
        items = []
        
        # Vol stats
        items.append(html.Div([
            html.Span("📊 Avg Vol: ", style={'color': '#aaa'}),
            html.Span(f"{regime.get('avg_volatility', 0):.2f}%", 
                     style={'color': '#ffa94d', 'float': 'right'})
        ], className="mb-2"))
        
        items.append(html.Div([
            html.Span("⚡ Max Vol: ", style={'color': '#aaa'}),
            html.Span(f"{regime.get('max_volatility', 0):.2f}%", 
                     style={'color': '#ff6b6b', 'float': 'right'})
        ], className="mb-2"))
        
        # Regime proportions with visual bars
        items.append(html.Div("📈 Regime Distribution:", 
                             style={'color': '#ffd43b', 'marginTop': '15px', 'fontWeight': 'bold'}))
        
        regime_colors = {
            'low': '#69db7e',
            'normal': '#4dabf7',
            'high': '#ffa94d',
            'extreme': '#ff6b6b'
        }
        
        regime_icons = {
            'low': '🟢',
            'normal': '🔵',
            'high': '🟠',
            'extreme': '🔴'
        }
        
        for regime_name in ['low', 'normal', 'high', 'extreme']:
            key = f'{regime_name}_proportion'
            if key in regime:
                prop = regime[key] * 100
                items.append(html.Div([
                    html.Div([
                        html.Span(f"{regime_icons.get(regime_name, '•')} {regime_name.capitalize()}: ", 
                                 style={'color': '#ccc'}),
                        html.Span(f"{prop:.1f}%", 
                                 style={'color': regime_colors[regime_name], 'float': 'right'})
                    ], className="mb-1"),
                    html.Div(style={
                        'height': '6px',
                        'width': f'{prop}%',
                        'backgroundColor': regime_colors[regime_name],
                        'marginLeft': '20px',
                        'marginBottom': '8px',
                        'borderRadius': '3px',
                        'boxShadow': f'0 0 8px {regime_colors[regime_name]}'
                    })
                ]))
        
        return html.Div(items, style={'fontFamily': 'monospace', 'fontSize': '13px'})
    
    def _create_hmm_params_panel(self, hmm_result: dict) -> html.Div:
        """Create HMM parameters panel."""
        if not hmm_result or 'error' in hmm_result:
            return html.P("No HMM data available", className="text-warning")
        
        items = []
        
        items.append(html.Div("🎯 HMM States:", 
                             style={'color': '#ffd43b', 'fontWeight': 'bold', 'marginBottom': '10px'}))
        
        for state in hmm_result.get('state_stats', []):
            color = ['#69db7e', '#4dabf7', '#ffa94d', '#ff6b6b'][state['state']]
            items.append(html.Div([
                html.Div([
                    html.Span(f"State {state['state'] + 1}: ", style={'color': '#aaa'}),
                    html.Span(f"Vol: {state['vol']*100:.2f}%", 
                             style={'color': color, 'float': 'right'})
                ], className="mb-1"),
                html.Div([
                    html.Span(f"  Proportion: ", style={'color': '#ccc'}),
                    html.Span(f"{state['proportion']*100:.1f}%", 
                             style={'color': color, 'float': 'right'})
                ], className="mb-1"),
                html.Div([
                    html.Span(f"  Mean Duration: ", style={'color': '#ccc'}),
                    html.Span(f"{state['duration']:.1f} periods", 
                             style={'color': color, 'float': 'right'})
                ], className="mb-2")
            ]))
        
        items.append(html.Div([
            html.Span("Regime Persistence: ", style={'color': '#aaa'}),
            html.Span(f"{hmm_result.get('persistence', 0)*100:.1f}%", 
                     style={'color': '#ffa94d', 'float': 'right'})
        ], className="mt-2"))
        
        return html.Div(items, style={'fontFamily': 'monospace', 'fontSize': '13px'})
    
    def _create_hmm_states_chart(self, session_data: pd.DataFrame, hmm_result: dict) -> go.Figure:
        """Create HMM states evolution chart."""
        if not hmm_result or 'states' not in hmm_result:
            return go.Figure()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Price
        fig.add_trace(
            go.Scatter(
                x=session_data.index,
                y=session_data['close'],
                name='Price',
                line=dict(color='white', width=1),
                hoverinfo='none'
            ),
            secondary_y=False
        )
        
        # HMM states as colored background
        states = hmm_result['states']
        state_colors = ['#69db7e', '#4dabf7', '#ffa94d', '#ff6b6b']
        
        for i, (start, end) in enumerate(self._get_state_segments(states)):
            if start < len(session_data) and end <= len(session_data):
                fig.add_vrect(
                    x0=session_data.index[start],
                    x1=session_data.index[end-1],
                    fillcolor=state_colors[states[start] % len(state_colors)],
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    secondary_y=False
                )
        
        fig.update_layout(
            template="plotly_dark",
            title="HMM Regime Evolution",
            xaxis_title="Time",
            height=400,
            showlegend=True,
            hovermode=False
        )
        
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_traces(hoverinfo='none')
        
        return fig
    
    def _get_state_segments(self, states):
        """Convert state array to segments."""
        if len(states) == 0:
            return []
        
        segments = []
        start = 0
        current_state = states[0]
        
        for i in range(1, len(states)):
            if states[i] != current_state:
                segments.append((start, i))
                start = i
                current_state = states[i]
        
        segments.append((start, len(states)))
        return segments
    
    def _get_current_regime(self, session_data: pd.DataFrame, session_info: dict) -> html.Span:
        """Get current volatility regime."""
        regime = session_info.get('regime_summary', {})
        if not regime:
            return html.Span("--", className="text-secondary")
        
        recent_vol = session_data['returns'].tail(20).std() * np.sqrt(252*390)
        avg_vol = regime.get('avg_volatility', 0)
        
        if recent_vol > avg_vol * 1.5:
            regime_name = "EXTREME"
            color = '#ff6b6b'
        elif recent_vol > avg_vol * 1.2:
            regime_name = "HIGH"
            color = '#ffa94d'
        elif recent_vol < avg_vol * 0.8:
            regime_name = "LOW"
            color = '#69db7e'
        else:
            regime_name = "NORMAL"
            color = '#4dabf7'
        
        return html.Span(regime_name, style={'color': color})
    
    def _detect_extremes(self, session_data: pd.DataFrame, session_info: dict) -> pd.DataFrame:
        """Detect extreme events."""
        stats = session_info.get('statistics', {})
        if not stats:
            return pd.DataFrame()
        
        prices = session_data['close'].values
        mean = stats.get('mean', 0)
        std = stats.get('std', 1)
        
        z_scores = (prices - mean) / std
        extreme_mask = np.abs(z_scores) > 2.5
        
        if not np.any(extreme_mask):
            return pd.DataFrame()
        
        extremes = []
        extreme_indices = np.where(extreme_mask)[0]
        
        for idx in extreme_indices:
            extremes.append({
                'time': session_data.index[idx],
                'price': prices[idx],
                'z_score': z_scores[idx],
                'type': 'High' if z_scores[idx] > 0 else 'Low'
            })
        
        return pd.DataFrame(extremes).sort_values('time')
    
    def _create_extreme_table(self, extremes_df: pd.DataFrame) -> html.Div:
        """Create extreme events table."""
        if extremes_df.empty:
            return html.P("No extreme events detected", className="text-warning")
        
        rows = []
        for _, row in extremes_df.head(10).iterrows():
            color = '#ff6b6b' if row['type'] == 'High' else '#4dabf7'
            rows.append(html.Tr([
                html.Td(row['time'].strftime('%H:%M:%S')),
                html.Td(f"${row['price']:.2f}"),
                html.Td(html.Span(row['type'], style={'color': color})),
                html.Td(f"{row['z_score']:.2f}σ")
            ]))
        
        table = dbc.Table(
            [html.Thead(html.Tr([
                html.Th("Time"), html.Th("Price"), html.Th("Type"), html.Th("Z-Score")
            ])),
            html.Tbody(rows)],
            className="table-custom",
            striped=True,
            bordered=False,
            hover=True,
            size="sm"
        )
        
        count_text = f"Total extreme events: {len(extremes_df)}"
        if len(extremes_df) > 10:
            count_text += " (showing first 10)"
        
        return html.Div([
            html.P(count_text, className="mb-2", style={'color': '#ffd43b'}),
            table
        ])
    
    def load_data(self):
        """Load and process data."""
        logger.info("Loading ES futures data...")
        
        # Fetch data with chunking to handle yfinance limits
        self.data = self.data_fetcher.fetch_data()
        
        # Create complete sessions
        self.sessions = self.session_manager.create_sessions(self.data)
        
        # Process each session
        for session_id, session_data in self.sessions.items():
            if not self.level_store.get_session(session_id):
                logger.info(f"Processing session {session_id}")
                
                # Compute statistics
                stats = self.statistics_engine.compute_session_statistics(session_data)
                
                # Fit distributions to prices
                dist_results = self.distribution_models.fit_all_distributions(
                    session_data['close'].values, is_returns=False
                )
                
                # Fit distributions to returns
                returns_data = session_data['returns'].dropna().values
                returns_dist_results = self.distribution_models.fit_all_distributions(
                    returns_data, is_returns=True
                )
                
                # Tail analysis
                tail_metrics = self.tail_analyzer.compute_tail_metrics(returns_data)
                
                # Regime analysis
                regime_summary = self.regime_detector.analyze_regimes(session_data)
                
                # HMM analysis
                if HMM_AVAILABLE and len(returns_data) > 50:
                    hmm_result = self.hmm_detector.fit_hmm(returns_data, session_id)
                    self.hmm_results[session_id] = hmm_result
                
                # Create levels dict
                levels = {
                    'mean': stats['mean'],
                    'plus_1sigma': stats['plus_1sigma'],
                    'minus_1sigma': stats['minus_1sigma'],
                    'plus_2sigma': stats['plus_2sigma'],
                    'minus_2sigma': stats['minus_2sigma'],
                    'plus_3sigma': stats['plus_3sigma'],
                    'minus_3sigma': stats['minus_3sigma'],
                    'p1': stats.get('p1', 0),
                    'p5': stats.get('p5', 0),
                    'p95': stats.get('p95', 0),
                    'p99': stats.get('p99', 0)
                }
                
                # Save to store
                session_info = {
                    'statistics': stats,
                    'distribution_models': dist_results,
                    'returns_distributions': returns_dist_results,
                    'tail_metrics': tail_metrics,
                    'regime_summary': regime_summary,
                    'levels': levels
                }
                
                self.level_store.save_session(session_id, session_info)
        
        logger.info(f"Loaded {len(self.sessions)} complete sessions")
    
    def run(self, debug: bool = False):
        """Run the dashboard."""
        self.load_data()
        
        import os
        port = int(os.environ.get("PORT", 8050))
        self.app.run(
            host="0.0.0.0",  # DŮLEŽITÉ: ne localhost!
            port=port,
            debug=False
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ES Futures Research Platform - Complete Sessions")
    parser.add_argument("--resolution", choices=["1m", "5m"], default="5m")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--days", type=int, default=30, help="Days of history to fetch")
    args = parser.parse_args()
    
    setup_logging()
    
    config.data_resolution = args.resolution
    config.dashboard_port = args.port
    config.debug_mode = args.debug
    config.use_caching = not args.no_cache
    
    # Check HMM availability
    if not HMM_AVAILABLE:
        logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")
    
    logger.info("=" * 70)
    logger.info("ES FUTURES QUANTITATIVE RESEARCH PLATFORM - COMPLETE SESSIONS")
    logger.info("=" * 70)
    logger.info(f"Resolution: {config.data_resolution}")
    logger.info(f"Port: {config.dashboard_port}")
    logger.info(f"Debug: {config.debug_mode}")
    logger.info(f"Caching: {config.use_caching}")
    logger.info(f"Days: {args.days}")
    logger.info(f"HMM Available: {HMM_AVAILABLE}")
    logger.info("=" * 70)
    logger.info("Sessions run from 18:00 to 17:00 next day (23 hours)")
    logger.info("=" * 70)
    
    try:
        dashboard = ESDashboard()
        dashboard.run(debug=config.debug_mode)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# GUNICORN ENTRY POINT
# ============================================================================

import os
import threading

setup_logging()

# Vytvoř dashboard BEZ načítání dat
_dashboard = ESDashboard()

def load_data_background():
    """Načte data na pozadí zatímco server už běží."""
    logger.info("Spouštím načítání dat na pozadí...")
    _dashboard.load_data()
    logger.info("Data načtena!")

# Spusť načítání dat na pozadí
_thread = threading.Thread(target=load_data_background, daemon=True)
_thread.start()

# Server musí být dostupný IHNED - tohle Gunicorn najde
server = _dashboard.app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    _dashboard.app.run(host="0.0.0.0", port=port, debug=False)